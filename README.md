# cpsc545-project

To reproduce all analyses, please run:

```
docker build -t myimg .
docker run -it -v ${PWD}:/home/rstudio/ myimg ./main.py --output_dir reproduced_output
```
