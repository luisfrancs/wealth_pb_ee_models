import zipfile
import struct
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def read_gt3x_file_with_fs(file_path):
    """
    Loads an ActiGraph GT3X file using stream_gt3x_to_parquet.
    A temporary Parquet file is created in the same directory as the
    original GT3X file and automatically deleted after loading.

    Args:
        file_path (str): Path to the .gt3x file.

    Returns:
        parquet_df (pd.DataFrame): Processed accelerometer data.
        hz (float): Extracted sampling frequency.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not .gt3x.
        RuntimeError: If processing fails.
    """
    try:
        # Validate input path
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("'file_path' must be a non-empty string.")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(  f"File does not exist: '{file_path}'"            )

        # Validate extension
        _, ext = os.path.splitext(file_path)

        if ext.lower() != ".gt3x":
            raise ValueError(  f"Unsupported extension '{ext}'. Only '.gt3x' is supported."    )

        # Generate temporary Parquet file in the same directory
        folder = os.path.dirname(os.path.abspath(file_path))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_parquet_path = os.path.join(folder, f"{file_name}_temp.parquet")

        try:
            # Convert GT3X to temporary Parquet
            output_file, hz = stream_gt3x_to_parquet(
                file_path=file_path,
                output_parquet_path=output_parquet_path)

            # Read processed data
            parquet_df = pd.read_parquet(output_file)
            return parquet_df, hz

        finally:
            # Remove temporary Parquet file
            if os.path.exists(output_parquet_path):
                os.remove(output_parquet_path)

    except (FileNotFoundError, ValueError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"ActiGraph processing error: {e}"
        ) from e


def stream_gt3x_to_parquet(file_path, output_parquet_path, chunk_records=1_000_000):
    """
    Streams and parses a .gt3x file in low-memory chunks, saving directly
    to a compressed Parquet dataset using PyArrow's native ParquetWriter.
    Returns: (output_parquet_path, sample_rate_hz)
    """
    metadata = {}

    # 1. First Pass: Read metadata text quickly
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('info.txt') as f:
            for line in f.read().decode('utf-8', errors='ignore').strip().split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    metadata[k.strip().lower()] = v.strip()

    scale = float(metadata.get('acceleration scale', 341.0))
    sample_rate_hz = int(metadata.get('sample rate', 100)) # This is your Hz value!
    start_ticks = int(metadata.get('start date', 0))

    # Setup absolute starting timeline
    unix_ts = (start_ticks - 621355968000000000) / 10_000_000
    current_time = pd.to_datetime(unix_ts, unit='s', utc=True)
    time_delta = pd.to_timedelta(1000 // sample_rate_hz, unit='ms')

    # Remove existing output file if it exists to avoid corruptions
    if os.path.exists(output_parquet_path):
        os.remove(output_parquet_path)

    # Define the PyArrow table schema explicitly (Timestamp Index + X, Y, Z columns)
    schema = pa.schema([
        ('Timestamp', pa.timestamp('ms', tz='UTC')),
        ('X', pa.float32()),
        ('Y', pa.float32()),
        ('Z', pa.float32())
    ])

    # 2. Second Pass: Stream binary file chunk-by-chunk using a persistent disk writer
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('log.bin', 'r') as f:
            with pq.ParquetWriter(output_parquet_path, schema, compression='snappy') as writer:

                chunk_buffer = []
                buffered_samples = 0

                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break  # End of file reached safely

                    if header[0] != 0x1E:
                        # Sync recovery: skip byte by byte until next sync marker
                        while True:
                            b = f.read(1)
                            if not b or b[0] == 0x1E:
                                header = b + f.read(7) if b else b
                                break
                        if len(header) < 8:
                            break

                    record_type = header[1]
                    payload_size = struct.unpack('<H', header[6:8])[0]

                    payload = f.read(payload_size)
                    f.read(1)  # Skip 1-byte checksum at the tail of each block

                    if len(payload) < payload_size:
                        break

                    # --- Decode Type 0 (12-bit Packed Legacy Format) ---
                    if record_type == 0:
                        b_matrix = np.frombuffer(payload, dtype=np.uint8)
                        b_matrix = b_matrix[:(len(b_matrix) // 3) * 3].reshape(-1, 3)

                        val1 = ((b_matrix[:, 0].astype(np.int16) << 4) | (b_matrix[:, 1] >> 4))
                        val2 = (((b_matrix[:, 1] & 0x0F).astype(np.int16) << 8) | b_matrix[:, 2])

                        val1 = np.where(val1 > 2047, val1 - 4096, val1)
                        val2 = np.where(val2 > 2047, val2 - 4096, val2)

                        flat_samples = np.empty(len(val1) + len(val2), dtype=np.int16)
                        flat_samples[0::2] = val1
                        flat_samples[1::2] = val2

                        samples_count = len(flat_samples) // 3
                        yxz_matrix = flat_samples[:samples_count * 3].reshape(-1, 3)
                        xyz_matrix = yxz_matrix[:, [1, 0, 2]]  # Reorder YXZ to XYZ
                        chunk_buffer.append(xyz_matrix)
                        buffered_samples += samples_count

                    # --- Decode Type 26 (16-bit Modern Format) ---
                    elif record_type == 26:
                        block_data = np.frombuffer(payload, dtype=np.int16)
                        samples_count = len(block_data) // 3
                        if samples_count > 0:
                            chunk_buffer.append(block_data[:samples_count * 3].reshape(-1, 3))
                            buffered_samples += samples_count

                    # --- Flush Buffer directly to Disk when threshold is met ---
                    if buffered_samples >= chunk_records:
                        matrix = np.vstack(chunk_buffer).astype(np.float32) / scale
                        chunk_times = pd.date_range(start=current_time, periods=len(matrix), freq=time_delta)

                        df_chunk = pd.DataFrame(matrix, columns=['X', 'Y', 'Z'])
                        df_chunk['Timestamp'] = chunk_times

                        table = pa.Table.from_pandas(df_chunk, schema=schema, preserve_index=False)
                        writer.write_table(table)

                        current_time = chunk_times[-1] + time_delta
                        chunk_buffer = []
                        buffered_samples = 0

                # Flush any remaining lines left in buffer right before closing file
                if chunk_buffer:
                    matrix = np.vstack(chunk_buffer).astype(np.float32) / scale
                    chunk_times = pd.date_range(start=current_time, periods=len(matrix), freq=time_delta)

                    df_chunk = pd.DataFrame(matrix, columns=['X', 'Y', 'Z'])
                    df_chunk['Timestamp'] = chunk_times

                    table = pa.Table.from_pandas(df_chunk, schema=schema, preserve_index=False)
                    writer.write_table(table)

    print(f"\nProcessing Complete! Outputs written to: {output_parquet_path}")

    # Return both items so your library knows what sample rate was inside the file
    return output_parquet_path, sample_rate_hz
