# Full-shot anime generation

**Which video models move convincingly, and a one-second repair**

A case study from Alvdansen Labs by Minta Carlson and Timothy Bielec · October 2026.

**Live site:** https://alvdansen.github.io/full-shot-anime-generation/

---

## Abstract

*Anime can now be generated a whole shot at a time, scene and motion together, from a single first frame. We compared seven video routes on six shots to find the ones that move convincingly with little work from an artist on the in-betweens, then regenerated one second of a failed take with our seq adapter.*

This study is about generating anime one full shot at a time. Each shot begins with a single generated frame, not layers or line art. A video model renders the whole scene from it: character, background and motion together. What we set out to find is which models move convincingly in this style with little direct work from an artist on the in-betweens, the drawings that carry a movement from one pose to the next. Anime usually holds each of those drawings for two frames, a rhythm called twos, which gives twelve drawings to every second of screen. Keeping that cadence, and keeping the motion believable, is what we judged every take on.

Some teams want methods built deliberately into a multi-stage animation pipeline, where an artist draws the keys and adapters trained on hand-drawn animation fill in the drawings around them. Our paper [Animating on Twos](https://alvdansen.github.io/animating-on-twos/) covers that approach. This case study is written for AI-native teams. It stays with shots generated whole, on tools a studio can open today, and asks how well each route keeps the cadence and the believability of the motion.

We gave seven routes the same character and the same six first frames, which we generated with Google's Nano Banana image models, working from hand-drawn references. A route here is a model together with the way it is run. Every route ran as a ComfyUI workflow on Floyo: ComfyUI is the node-graph tool in which most open video pipelines are built, and Floyo runs ComfyUI workflows in the cloud. The seven routes are:

- Four commercial models called through Floyo's partner nodes: MiniMax H3 API, Seedance 2.5, FLUX 3 and Wan 3.0
- MiniMax H3, run from its open weights
- Two adapters we trained on top of those weights, the seq adapter and the hero adapter

Each route animated each first frame twice, into a 5-second shot. Our Creative Lead, Minta Carlson, graded every take by eye, and we measured every take frame by frame. Then we set out to repair one take that went wrong. Its knock hits the peephole lens instead of the door, so we regenerated that second with our seq adapter as a tween to cut back into the take.

---

## Repository contents

| File | Description |
| --- | --- |
| [`index.html`](index.html) | The case study as a web page: what the live site serves. |
| [`case-study.md`](case-study.md) | The same article as Markdown, readable on GitHub. |
| [`media/`](media/) | 17 videos and 26 images: the comparison grids and their posters, the stills, and the Part B clips and figures (79 MB). |
| [`workflows/`](workflows/) | Four ComfyUI workflows in API format, with a README. |
| [`build_site.py`](build_site.py) | The script that generated the page, the Markdown, the media and the workflows. |
| [`LICENSE`](LICENSE), [`LICENSE-CODE`](LICENSE-CODE) | The licenses (see below). |

The script records how the page was built. Its inputs (the generated clips, grade sheets, run records and graph builders) stay in our
working directory.

---

## Citation

```bibtex
@misc{full_shot_anime_generation_2026,
  title        = {Full-shot anime generation: which video models move convincingly, and a one-second repair},
  author       = {Carlson, Minta and Bielec, Timothy},
  year         = {2026},
  month        = {October},
  howpublished = {Case study, Alvdansen Labs},
  url          = {https://alvdansen.github.io/full-shot-anime-generation/}
}
```

---

## License

- **Article text, figures and video:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). You may copy, redistribute,
  remix and build on the work in any medium or format, including commercially, provided you give appropriate credit and link to the
  license.
- **Code** (`build_site.py` and the workflow files in `workflows/`): [MIT](LICENSE-CODE).
