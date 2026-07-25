# Models

This directory holds models you train locally. It is empty in a fresh clone.

After training through the GUI, the application writes a schema-marked v2
portable bundle and a same-environment pickle here, and **Load Last Model**
then reads them from this directory.

The released reference classifier is not stored here. It ships in
`released_model/`, and **Load Last Model** falls back to it whenever this
directory contains no locally trained model, so training never overwrites the
released bundle.

Generated model files are ignored by Git.
