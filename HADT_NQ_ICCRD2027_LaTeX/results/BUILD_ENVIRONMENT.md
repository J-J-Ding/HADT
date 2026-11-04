# Manuscript build environment

The manuscript uses the supplied `acmart.cls` (ACM class v1.71) and
`ACM-Reference-Format.bst`. A user-local Tectonic 0.17.0 Linux x86-64 musl
binary supplies XeTeX, BibTeX, LaTeX packages, and fonts without a system TeX
installation. No system packages or administrator permissions are needed.

## Build

From the project root:

```bash
./scripts/build_paper.sh
```

The script retains `main.log`, bibliography output, and intermediate files
beside `main.pdf`. It automatically performs bibliography processing and
the required LaTeX reruns. It can also compile an isolated copy and place
all generated files in a separate output directory:

```bash
./scripts/build_paper.sh /tmp/hadt_tex_verification/main.tex /tmp/hadt_tex_build
```

The first build downloads packages from Tectonic's pinned v33 resource
bundle. Downloaded resources and the LaTeX format are cached in
`~/.cache/tectonic/`. An already cached document can be rebuilt offline:

```bash
TECTONIC_OFFLINE=1 ./scripts/build_paper.sh
```

New packages or fonts may require one online build. The tool path can be
overridden with `TECTONIC_BIN=/path/to/tectonic`.

## Reinstall the engine

```bash
mkdir -p .tools/tectonic
curl -L --fail --retry 2 -o /tmp/tectonic-0.17.0.tar.gz \
  'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
echo '8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7  /tmp/tectonic-0.17.0.tar.gz' | sha256sum -c -
tar -xzf /tmp/tectonic-0.17.0.tar.gz -C .tools/tectonic
.tools/tectonic/tectonic --version
```

Downloaded archive SHA-256:
`8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`.
Extracted binary SHA-256:
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`.
Resource bundle: `https://relay.fullyjustified.net/default_bundle_v33.tar`.
Bundle content identifier:
`6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`.

## Verification

Initial compile verification is performed against an isolated copy of the
original template in `/tmp/hadt_tex_verification/`; the active manuscript,
bibliography, and figures are not changed by toolchain setup.

Verified on 2026-10-30:

- Tectonic 0.17.0 successfully compiled the preserved original manuscript,
  including BibTeX and cross-reference reruns.
- `TECTONIC_OFFLINE=1 ./scripts/build_paper.sh
  /tmp/hadt_tex_verification/main.tex /tmp/hadt_tex_script_verification`
  succeeded after the first online build, proving the supplied script and
  cached toolchain work without further network access.
- The resulting original manuscript has 7 letter-size pages and anonymous
  author metadata; all fonts are embedded.
- The original figure PDFs contain embedded Type 3 DejaVuSans fonts. For
  regenerated Matplotlib figures, set `matplotlib.rcParams['pdf.fonttype'] =
  42` before exporting to avoid Type 3 fonts in final submission artwork.
- No unresolved citations, references, missing characters, or fatal TeX
  errors remain in the final original-template log. BibTeX reports missing
  publisher/address/pages fields in four original entries; those are
  bibliography metadata warnings rather than compilation failures.
- The original source produces overfull text boxes, with the largest
  26.59 pt, and the ACM class warns that its reference-format block is
  disabled by `printacmref=false`. Review those choices in the final paper.
  `cmap: pdftex not detected` is expected with the XeTeX engine.

The verification PDF and full diagnostic files remain in
`/tmp/hadt_tex_script_verification/`. Builds are reproducible in tool and
resource versions; PDF creation timestamps may differ between runs.
