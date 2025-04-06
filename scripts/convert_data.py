import os
import sys

import h5py
import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


def convert_scene_to_folder(h5_file_path, output_dir):
    scene_name = os.path.basename(h5_file_path).split('.')[0]
    scene_dir = os.path.join(output_dir, scene_name)

    image_dir_blur = os.path.join(scene_dir, 'blur')
    image_dir_sharp = os.path.join(scene_dir, 'sharp')
    events_dir = os.path.join(scene_dir, 'events')

    os.makedirs(image_dir_blur, exist_ok=True)
    os.makedirs(image_dir_sharp, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)

    with h5py.File(h5_file_path, 'r') as f:
        # Extract image data
        blur_images = sorted(f['images'].keys())
        sharp_images = sorted(f['sharp_images'].keys())

        # Extract event data
        xs = f['events/xs'][:]
        ys = f['events/ys'][:]
        ts = f['events/ts'][:]
        ps = f['events/ps'][:]

        # Total number of frames in the scene
        num_frames = len(blur_images)

        for idx, (blur_img, sharp_img) in enumerate(zip(blur_images, sharp_images)):
            blur_image_data = f['images'][blur_img][:]
            sharp_image_data = f['sharp_images'][sharp_img][:]

            # Convert BGR to RGB
            blur_image_data = blur_image_data[..., ::-1]
            sharp_image_data = sharp_image_data[..., ::-1]

            blur_image = Image.fromarray(blur_image_data)
            sharp_image = Image.fromarray(sharp_image_data)

            blur_image.save(os.path.join(image_dir_blur, f'{scene_name}_blur_{idx:05d}.png'))
            sharp_image.save(os.path.join(image_dir_sharp, f'{scene_name}_sharp_{idx:05d}.png'))

            # Get the timestamp range for this image frame
            t0 = f['events/ts'][0]
            tk = f['events/ts'][-1]
            start_time, end_time = get_frame_time_window(idx, t0, tk, num_frames)

            # Find events that correspond to this frame's time window
            filtered_xs, filtered_ys, filtered_ts, filtered_ps = filter_events(xs, ys, ts, ps, start_time,
                                                                                    end_time)

            event_data = np.column_stack((filtered_xs, filtered_ys, filtered_ts, filtered_ps))

            np.savetxt(os.path.join(events_dir, f'{scene_name}_events_{idx:05d}.txt'), event_data, fmt='%d %d %d %d')

    print(f"Converted {h5_file_path} into {scene_dir}")


def get_frame_time_window(index, t0, tk, num_frames):
    # Duration in which the whole scene was shot is the timestamp of the last event - time stamp of first event
    total_time_duration = tk-t0
    # Assume each frame has equal duration
    frame_duration = total_time_duration // num_frames
    # Start and end timestamps for the current frame
    start_time = index * frame_duration
    end_time = (index + 1) * frame_duration

    return start_time, end_time


def filter_events(xs, ys, ts, ps, start_time, end_time):
    # Filter the events based on the time range, to get the events corresponding to one frame
    mask = (ts >= start_time) & (ts < end_time)
    filtered_xs = xs[mask]
    filtered_ys = ys[mask]
    filtered_ts = ts[mask]
    filtered_ps = ps[mask]
    return filtered_xs, filtered_ys, filtered_ts, filtered_ps


def run():
    h5_files_train = [f for f in os.listdir("data/GOPRO_rawevents/train") if f.endswith('.h5')]
    output_dir_train = 'data/GOPRO_converted_2/train'
    for h5_file_train in h5_files_train:
        convert_scene_to_folder(os.path.join("data/GOPRO_rawevents/train", h5_file_train), output_dir_train)

    h5_files_test = [f for f in os.listdir("data/GOPRO_rawevents/test") if f.endswith('.h5')]
    output_dir_test = 'data/GOPRO_converted_2/test'
    for h5_file_test in h5_files_test:
        convert_scene_to_folder(os.path.join("data/GOPRO_rawevents/test", h5_file_test), output_dir_test)


