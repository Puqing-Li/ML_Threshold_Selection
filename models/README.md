# Models

This directory intentionally contains no classifier.

After training through the GUI, the application writes a schema-marked v2
portable bundle and a same-environment pickle here. **Load Last Model** becomes
available only after that training has completed successfully.

Generated model files are ignored by Git. A release model should be added only
after its training inputs, feature schema, and practical TomoFab checks have
been frozen.
