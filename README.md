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

After this, you get dataset/etl9g.pkl as training dataset file.

### Train DCGAN model

```
$ python src/train_dcgan_48px.py -g 0 -o model/dcgan_48px --out_image_dir image/dcgan_48px
```

## License

MIT
