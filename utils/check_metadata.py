import os
import json
from pathlib import Path
from datetime import datetime

import pdb

cwd = Path(os.getcwd())
metadata_files = [item for item in cwd.rglob("*.json") if item.parent != cwd]

total_time = 0
total_ad_frames = 0
total_content_frames = 0
total_transition_frames = 0
for metadata_file in metadata_files:
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        total_ad_frames += metadata["total_ad_frames"]
        total_content_frames += metadata["total_content_frames"]
        total_transition_frames += len(metadata["transitions"])

        content_length = metadata["content"]["length"]
        try:
            parsed_time = datetime.strptime(content_length, "%H:%M:%S.%f")
        except ValueError as e:
            parsed_time = datetime.strptime(content_length, "%H:%M:%S")
        num_seconds = parsed_time.hour * 3600 + parsed_time.minute * 60 + parsed_time.second
        total_time += num_seconds

total_frames = total_ad_frames + total_content_frames + total_transition_frames
print(f"Total hours of data collected: {total_time / 3600}")
print(f'''
    AD FRAMES {round(100 * total_ad_frames / total_frames, 1)}%,
    CONTENT FRAMES {round(100 * total_content_frames / total_frames, 1)}%,
    TRANSITION FRAMES {round(100 * total_transition_frames / total_frames, 1)}%
''')