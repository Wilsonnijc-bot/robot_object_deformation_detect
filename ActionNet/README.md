# ActionNet small sample

Browse the fifteen clips in the [dataset viewer](https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/ActionNet/). The [CSV manifest](metamanifest/world_model_prompts.csv)
lists each conditioning image, action prompt, reference video, and review.

Source: [FourierIntelligence/ActionNet](https://huggingface.co/datasets/FourierIntelligence/ActionNet),
revision `eb3b5bb16f3a5a7052c1498c1072e78ac99042d0`.
Publisher license: **CC BY-NC-SA 4.0**. Source access requires an authorized
Hugging Face account; the existing browser session was used for this download.
No access token is stored here.

The first three episodes are a convenience sample of **the same croissant-placement
task**, from the smallest archive, `01JN0Y324H-01JN0YMV54.tar` (1,067,192,320 bytes).
It is not a sample of three distinct tasks or a representative sample of ActionNet.
The selected episodes are the first, the episode with a different source caption,
and the last episode in that archive. The publisher camera is named `top`; its
close view shows the hands and work surface rather than the full humanoid body.

| Episode | Duration | Frames | Resolution |
| --- | ---: | ---: | --- |
| `01JN0Y324H2ZFDTSYD` | 14.40 s | 432 | 1280 × 800 |
| `01JN0YDDF5ZYHXSSFO` | 18.60 s | 558 | 1280 × 800 |
| `01JN0YMV54BPXU4PLB` | 16.37 s | 491 | 1280 × 800 |

## Twelve additional episodes

Twelve more episodes were added from four other archives (`01JGJCNDDF-01JGJY03Q1.tar`,
`01JHF7ZEBF-01JHFAFHR7.tar`, `01JJ130F0A-01JJ15BREB.tar`, `01JMC4WT0H-01JMCBC79M.tar`;
about 4 GB each), same repository revision and license. The archives are plain `.tar`
files, so `scripts/fetch_more.py` walks the tar headers with small HTTP Range requests and downloads
only the four members of each selected episode (about 500 MB in total instead of ~17 GB).
Three episodes with different source prompts were taken from roughly the 10%, 50% and 90%
positions of each archive. This is again a convenience sample, not a representative one;
the extraction is byte-exact and each member's SHA-256 is in `metamanifest/metadata/download.json`.

| Episode | Source task | Publisher caption (verbatim) | Frames |
| --- | --- | --- | ---: |
| `01JGJQAJP3GQBKPCED` | pnp | Pick up the cup from the table and place it in the container | 418 |
| `01JGJS4GWCL7A5XCXC` | pnp | Pick up the orange from the table and place it in the container. | 404 |
| `01JGJXRGT7E3ULU66F` | shelf | Pick up the bread and place it on the first level of the double shelf. | 420 |
| `01JHF84WJCMA7S2S3S` | shelf | Take the lemon and put it on the second shelf | 461 |
| `01JHF8YGPVVKAXY7FK` | shelf | Take the croissant and put it on the second shelf | 421 |
| `01JHFA7YZJTTECHOFB` | pnc | Pick up the tape measure,place it into the box,and close the door. | 395 |
| `01JJ136C6DH2EZB42B` | npnp | Pick up the orange and place it into the container. | 410 |
| `01JJ141D4HW47GWFAL` | npnp | Pick up the clamp from the table and place it in the container. | 396 |
| `01JJ15873JBIEVYT62` | npnp | Pick up the clamp and place it into the container. | 405 |
| `01JMC54RMAVQ5E7IKP` | npnp | Pick up the Rubiks cube and place it into the transparent container. | 429 |
| `01JMCA5VKE4FOEQ7FA` | npnp | Move the tennis ball into a clear plastic box | 401 |
| `01JMCB8Z194OS4IEC5` | npnp | Move the clock and place it into the transparent container. | 445 |

All twelve are 1280 × 800 at 30 FPS, kept complete and unmodified (the generation video is the
source `top/rgb.mp4` itself), with a dedicated reviewer per video and prompts following `humanoid_action_v1`.
The publisher captions do not always match the footage, and each review records the
discrepancy. Two clear cases: `01JMCB8Z194OS4IEC5` is captioned "clock" but shows a golden
pastry, and `01JMCA5VKE4FOEQ7FA` is captioned "tennis ball into a clear plastic box" but shows a
tape measure placed on a dish rack. "Container" is often a basket, bowl or plate. In every
episode the active hand is described by the image edge it enters from, because the torso is
not visible and no robot-relative side could be established. Several clips show a person in the
background who never touches the robot or objects.

Each episode follows the existing dataset organization:

```text
<episode_id>/
├── original/
│   ├── <episode_id>.hdf5
│   ├── task_prompt.txt
│   ├── top/
│   │   ├── rgb.mp4
│   │   ├── depth.mkv
│   │   └── timestamps.json
│   └── metadata/metadata.json
└── generation input/
    ├── first_frame.png
    ├── actiongeneration_prompt.txt
    └── metadata/
        ├── metadata.json
        └── action_review.json
```

All source members are extracted unchanged. The generation video is the complete
source `original/top/rgb.mp4` itself (`relative_video_path` in the generation metadata);
no separate copy is kept, and there is no trimming, cropping, resizing, frame-rate
conversion, or video re-encoding. Each PNG is decoded from
source frame zero at native resolution and has no duplicate under `original/`.
Per-episode metadata paths resolve from the metadata JSON directory.

Each episode receives a dedicated visual review of the complete sequence and its
initial frame. Prompts follow `humanoid_action_v1` to preserve relevant hand roles
and actual outcomes; anatomical sides are used only when visually supported.
Detailed chronology, contact evidence, uncertainty, and source-caption differences
are retained in the review JSON. Inspection images are temporary, outside this
dataset folder.

Full RGB and depth decoding passed for all three episodes. Generation videos
are the original files; PNG RGB pixels match source frame zero.
HDF5 signatures were checked; robot arrays have not been semantically validated.
Checksums, source archive members, source annotations, and the archive checksum
are retained under per-episode metadata and `metamanifest/metadata/`.

To reproduce the first three episodes from the same downloaded archive and source annotation JSON:

```sh
../.venv/bin/python scripts/prepare_sample.py /path/to/01JN0Y324H-01JN0YMV54.tar /path/to/metadata.json
```

This recreates extraction metadata. Visual reviews and authored prompts are separate
artifacts. After reviews are present, rebuild the manifest with:

```sh
../.venv/bin/python scripts/build_manifest.py
```

The twelve additional episodes are fetched and built with:

```sh
HF_TOKEN=... ../.venv/bin/python scripts/fetch_more.py /path/to/new_work_directory
../.venv/bin/python scripts/prepare_more.py /path/to/new_work_directory
../.venv/bin/python scripts/build_manifest.py
```

The token is read from the environment only and is not stored. Authored prompts and reviews
are separate artifacts that must be present before the manifest is rebuilt.
Run `../scripts/quality_gate.py` first so the manifest carries the sharpness-gate columns
(all fifteen pass, first-person, 364–889).
