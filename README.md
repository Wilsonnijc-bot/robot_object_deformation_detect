# Robot video data branch

This branch contains the Robotvideo dataset project: TAVP, Humanoid Everyday,
NVIDIA GR1, source annotations, visual reviews, and preparation scripts.
It has an independent history on `data/robotvideo` in
`Wilsonnijc-bot/robot_object_deformation_detect`.

Videos and PNG images use Git LFS. To retrieve the complete data:

```sh
git lfs install
git clone --branch data/robotvideo --single-branch https://github.com/Wilsonnijc-bot/robot_object_deformation_detect.git Robotvideo
cd Robotvideo
git lfs pull
```

For TAVP generation inputs, read `TAVP/world_model_manifest.csv`. Its relative
video, first-frame, and prompt paths select the preferred reference for each
of the 23 source videos. Ten references use `world_model/first_substep/`;
thirteen use `world_model/` directly. Preserve those paths when uploading.
See [TAVP/WORLD_MODEL_README.md](TAVP/WORLD_MODEL_README.md) for input semantics
and validation details, and [AGENTS.md](AGENTS.md) for the prompt protocols.

Humanoid Everyday third-person and NVIDIA GR1 generation inputs use
`generation prompt/` beside `original/`. Each generation folder contains
`actiongeneration_prompt.txt`, `first_frame.png`, `video.mp4`, `action_review.json`,
and `metadata.json`. There is no duplicate `prompt.txt` in these two datasets.
Their `metamanifest/world_model_prompts.csv` files select the current input paths.
TAVP retains its existing `world_model/` layout and prompt filenames.

This commit is a snapshot of the local project. Humanoid and GR1 review files
may be updated by ongoing review work; their presence alone does not establish
that the latest protocol review is complete. Dataset sources and original
annotations remain alongside the generation references.

The local Python environment, caches, incomplete downloads, and credentials
are excluded. Consult each dataset's source metadata for provenance.
