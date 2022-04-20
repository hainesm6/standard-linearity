# Usage

Here's an example of using `standard-linearity` on the command line to iterate through a collection of csv files in a directory. The name of the output directory is dynamically created:

```shell
$ for f in ./*; do fbase=$(basename -s .CSV $f); standard-linearity -f WLS -h 3 -n 27 -o $fbase -p [path-to-config-file] -r " Blank corrected based on Raw Data (F: 482-16/525-20)" --skip-rows 3 -s "Standard Concentrations" $f; done
```
