# wav2vec-burn

An implementation of Meta's [wav2vec 2.0](https://arxiv.org/abs/2006.11477) speech transcription using the [Burn ML Framework](https://github.com/tracel-ai/burn).

This crate is a work in progress, and does not have a stable API yet.

## Testing

Some tests in this repo require the test data in [`test-data/`], which is retrievable with [`git-annex`](https://git-annex.branchable.com/). First, install `git-annex`, and then run:

```sh
git config annex.private true
git annex get test-data
```

Some files under `test-data/` are only available via a `git-annex` "special
remote". For example, to download from GitHub LFS:

``` sh
git annex enableremote github
git annex get test-data
```

All tests should now complete when run. Consider running tests in `--release` mode, as `burn` runs very slowly with optimizations disabled.

``` sh
cargo test --release --workspace
```

## Contributing Bug Reports

GitHub is the project's bug tracker. Please [search](https://github.com/privacyresearchgroup/wav2vec-burn/issues) for similar
existing issues before [submitting a new one](https://github.com/privacyresearchgroup/wav2vec-burn/issues/new).

# License

Licensed under [MIT](https://opensource.org/licenses/MIT).
