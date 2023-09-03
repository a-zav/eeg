# -*- coding: utf-8 -*-

import time
import numpy as np

from mne_realtime import LSLClient, MockLSLStream


def classify_from_stream(model, eeg_data):
    stream_id = "eeg-data-stream"

    eeg_data_stream = MockLSLStream(host=stream_id,
                                    raw=eeg_data,
                                    ch_type='eeg',
                                    status=True)

    client = LSLClient(info=eeg_data.info, host=stream_id)
    
    try:
        buffer = None

        eeg_data_stream.start()
        client.start()

        it = client.iter_raw_buffers()
        start_time = time.perf_counter_ns()
        last_update_time = start_time

        while True:
            data = next(it)
            if len(data) == 0:
                time_since_last_update = time.perf_counter_ns() - last_update_time
                if time_since_last_update > 1e9:
                    print(f"[{time.perf_counter_ns() - start_time}] No data, exiting...")
                    break
                continue
            
            last_update_time = time.perf_counter_ns()
            
            if buffer is not None:
                buffer = np.append(buffer, data, 1)
            else:
                buffer = data

            if buffer.shape[1] <= 160 * 0.5:
                #print(f"{time.perf_counter_ns() - start_time} " +
                #      f"buffer.shape = {buffer.shape}, waiting for more data...")
                continue

            print(f'[{time.perf_counter_ns() - start_time}] Start')
            result = model.predict(buffer[:,:81].reshape(1, 64, 81))
            print(f'[{time.perf_counter_ns() - start_time}] End, result = {result}')
            buffer = buffer[:,16:]
    finally:
        client.stop()
        eeg_data_stream.stop()

