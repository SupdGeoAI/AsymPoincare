# Dataset archive

The raw `datasets/` directory is stored as split ZIP parts to stay below GitHub's
single-file size limit.

To restore it after cloning:

```bash
cat data_archives/datasets.zip.part-* > datasets.zip
unzip datasets.zip
rm datasets.zip
```

