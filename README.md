# vcrunch

vcrunch is a containerised utility for remuxing and compressing video sources into AV1 Matroska files. The script is designed to run inside a Podman container and orchestrates `ffmpeg`, `svt-av1`, and `mkvmerge` to dump streams, encode video, and assemble the final output.

## Repository layout

- `Containerfile` – build definition for the runtime image.
- `vcrunch.py` – main entrypoint for the tool.
- `spec.md` – Gauge specification describing expected behaviour.
- `tests/` – unit tests for the script.

## Development

Set up a virtual environment and install the tooling you need for development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install pre-commit pytest
```

Run the automated checks before submitting changes:

```bash
pre-commit run --all-files
pytest
```

Build and run the container image locally to exercise the full workflow:

```bash
podman build -t vcrunch .
podman run --rm -v "$PWD:/workspace" vcrunch --help
```

## Release

Cut a semantic version tag to publish a new container image to GitHub Container Registry:

```bash
git tag -a v1.2.3 -m "v1.2.3"
git push origin v1.2.3
```

The release workflow builds multi-architecture images and pushes them to `ghcr.io/nashspence/vcrunch` with the tag name, `latest`, and the Git commit SHA.
