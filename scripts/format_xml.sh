#!/bin/bash
find . -name \*.xml -print0  | xargs -n1 -0 xmlformat --indent 1 --indent-char "        " --eof-newline --overwrite
