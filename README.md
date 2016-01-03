# Deep learning script for ETL9G using Chainer

## Requirement

* Python
* [Chainer](http://chainer.org/)
* [Pillow](https://pillow.readthedocs.org/en/3.0.x/)

## Usage

### Download dataset

Download ETL9 dataset from http://etlcdb.db.aist.go.jp/.  
You need registration to download.  
After registration, download ETL9G.zip, extract it, and put ETL9G_nn files into dataset directory.

### Convert dataset

```
$ python convert_dataset.py dataset dataset
```

It takes about 10 minutes.
After this, you get dataset/etl9g.pkl as training dataset file.

### Train DCGAN model

```
$ python src/train_dcgan.py -g 0 -o model/dcgan --out_image_dir image/dcgan
```

It takes about 25 minutes for 1 epoch on GTX 970.

### Generate image using DCGAN model

```
$ python src/generate_dcgan.py -m model/trained.gen.model -t あいうえお -c utf-8 -o image/gen.png
```

## Generated image sample

Generated after 50 epoch training  
<img src="https://raw.githubusercontent.com/dsanno/chainer-etl9/master/image/training_sample.png" width="640px" alt="generated image sample">

## License

MIT
