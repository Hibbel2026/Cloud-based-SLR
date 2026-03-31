from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role="YOUR_ROLE",
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.0",
    py_version="py310"
)

estimator.fit({
    "train": "s3://hiba-slr-filer/data_skeleton/train",
    "val": "s3://hiba-slr-filer/data_skeleton/val",
    "test": "s3://hiba-slr-filer/data_skeleton/test"
})