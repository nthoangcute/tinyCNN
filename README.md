# tinyCNN
> Very simple convolutional neural network by Java
## INSTALLING
Download the MNIST or EMNIST dataset from ![Google Drive](https://drive.google.com/drive/folders/10MfF2F5M40NxEFLSpaHWCMo4y8yEMivI?usp=sharing).

Change your path in **trainFile** and **testFile**. 
```
  public String trainFile = "/home/vietbt/java/mnist_digits_train.txt";
  public String testFile = "/home/vietbt/java/mnist_digits_test.txt";
  public double learningRate = 0.55;
  public int batchSize = 50;
  public int outputSize = 10;
```
You also can config the *learningRate*, *batchSize* or *outputSize* to match your dataset.
More information about MNIST or EMNIST datasets is in ![here](https://www.nist.gov/itl/iad/image-group/emnist-dataset). 
After that, run this code with your Java IDE or by linux command line:
```
  javac tinyCNN.java
  java tinyCNN
```
## PERFORMANCE
This program runs with only 1 CPU, so it takes a long time to get the best accuracy.

### MNIST Digits Dataset

* Best test accuracy: 99.13% with learning rate = 0.55 after 142,500 steps

<p align="center"><img src="https://lh3.googleusercontent.com/SUuX04Bf4Zyou0I1SNztWkSUEgX7Kz4_5fBtETnj1g6e5ClQIkXHSFwYj96BpXSwLkrbeRZiJcZDSHqGfh2yrNLYzfvyCuqc1G5XRarmIkfmXlZd0uiCiRThgLl8aK77AZFHKb0LfrZkztd0sQRHgPsD1KzHDlxHHUitfgHO49ul72DsbG1lMOmblDISUEGXE1BwIWEHOvnPmQnn-QhzMvupV5oe8N-S70A2pg79lTUXhonwCQb7xYYa8QfL7SLvnW57si1eTHJoWaG9rKfqMl5MUyVbHsBv3BfJssfloGHNunhknXpR29aOwDGPNYWL_jnqjU5QiGyb_QOfVAtUyAsUqdO_AN2MbpbjIkxSlsN_nYlCSN0zNnTH6mmfyPtNQoq1NGbJV_jIOo_qi43DDRc_PZETAtU9XkJ8uy_aQb9JHTqzSVFfmnnL9-7AY_w3h4zFrOsAhllX6MdWu3or93BJiyFwGlj1AikGAaqBH792d_zPa2Xs6S--0bS_YLZTnTVCanfN3r-Ze4TIu7fNWJr6GyNmcth0KZVCpTHGa72etG_SyYZmqIumGyRI4l3D4Kf2WfYo_vmk13NKzZruX60XIaP1N0vCgqZfBq0=w1395-h649-no" width="600"></p>

### EMNIST Digits Dataset

* Best test accuracy: 98.58% with learning rate = 0.87 after 60,200 steps

<p align="center"><img src="https://lh3.googleusercontent.com/IsW3x5EwfPlhAODR8I-JRSVDZ4xUAyqzElbnr__3KvOSimwI4QAbWP8kmAA3fCfCVbD5Y4e8EBfgPqPFqk8Vki9DVLJUllOz0gMiLF8moHhysFusqcKGNLhO8ef2TX9wvbrO5S0T28r9Wr8lm99_L9QhBY7alYwH4l8ImeC3TIEpSC6IzYs79QMAbwdfA6rMWYcG-AGkEcoPV2xASZf5zr1EQrofRUOeM3kxMLhdilv7vxBnmcqKkzqrSIslLUDM7Y9Cnxx2aVqytW-LFmEqNxfSaS1Ap1f77UucIAcpJO-xaSCzJEHL9JdQqGCxuoSGpL1oOEEWoM9O6B9XgmHoJUFbkhMdDNltk0P-cb8SLQPRgxnHSxIjDLGqt0Ce9K1uULMojOOu_VTVj2p_EWqcIg3YrRF-5KR0iobRHbpNK5dVNc_oXIfvIQRWyKWZtr-vf2mhal38NC7pVCPRz0EVYBc9ewg_1b4LfNeoDRXRM_DaeljlFbEUHut6WX2qlw7FBHV5bgyDJIA4Uk6i3lvSf8-85qfa3zLBk-WbovMUPmhczKVzipES10lIZN0_RUzJiA3VlFjiEtdgM_73qrP1P8DbKYdMm3rZOMPea3U=w1257-h651-no" width="600"></p>

### EMNIST Letters Dataset

* Best test accuracy: 88.56% with learning rate = 0.6 after 63,100 steps

<p align="center"><img src="https://lh3.googleusercontent.com/Aojesd2H9s3UnrQUX14juc823uYS_VLdvQfP6N58L2M-maxaj5Zk2dxeGt3oz6nMlv_nyDI4dyTJiIX99Ghygp4R69oDXyO2im47Gja7wpgj4hPvRW--zbmgnSxOp7NYwfvd9sAtgAtCK5DcSNzBZzU2_1XL1bptKSAuWRjGBl42_5RL4vR6XnDm9uKEPEUvpD0-7yAQ7WgWdUcwbeQRdAS7GDnK-OIbht6XC9OBMdhrhJlFNoDJjMwI7LRrxdybRuKGzM28092BzYmiwlF21uE5ibfxPFZ61WwaTbT4dOQvwPK_CciBy54RErHFKc5wDAvczSXWswiUZIebNA4Xd2Y81gU-HuXEZOyp32YU49Wu5N7HEmp1KKk7Y-NdEMmo0qAvPu3_657rX5uk2ubgeM0D2PIea5KB2zpSsLtS3jj51ipFAP9IARH-rNRgzX8NvvtKNpd74yj6zj9_-4d6ZmGm0KId6jGMAcmpNGJXzYkinQKG2OH_HGpyILZ1p1e5uwyGiicLA1vmkQ9XFcvFjGlXz-8cAKJJynMKuXWskiaCdvP0Ed0SYdrqkTydKgBb6_rNq2qsb_YpR03EnXshRR-vuiHs5k14BBI1YpI=w1261-h695-no" width="600"></p>

### EMNIST Balanced Dataset

* Best test accuracy: 83.81% with learning rate = 0.6 after 117,300 steps

<p align="center"><img src="https://lh3.googleusercontent.com/6eu2ed50adGMy1geUp6ajNrXxBpoBgatw_QsgS2kAN74nQBGnE8h_hSf8VnVDr_S2rQambdMuM_bkNbKAWuxptARwZEDAUgYLZ0uGX3N4F4G4raY2OBcjTWw4ZRfPusYmybZe7SQjIUwVZAt3rHgDTeBpT6xpUiS3jEF2DS0q1dlwqZaiZVDgQQUcYuarXi6crSbYMBk2DaNn1-KqtXJWDE5UsSHtVRo1nLNXEeSjFXKdj5D7a6ZWqmCTxJnDjwqRxVJn4lZO4xwgIjhXjXH57Sn-4_XmDGVBLeuPSistxHH9TzPa2iNl8juo8d3XihYjSG_0gHPTTG5w-3PnAa6WXK7Lm5J6gAwSbtvqYoDNInQkpKpjRfYSWPUS3lemvg1Z-wVx2ibUYa0JceBzKLElaXh_8sxx-wB28gfxc8XwSSaIAAnkRaMqo7MLujKHI25fYv-ZWO3uO_qSdi6QZYVWJdOoFUadOl-IiDTee1ah2Qe3TghS1uRyvRXPYKdhpZ6mKY4dtAyDIl8Irke_MI-ezPnnaQNvdAkQUCXVgBmv8xpzPuNncqQi4esZjVZ83lmUueueYbTmYqzEnhdtuFD9EzxQwHsm6jFbVhePDc=w1261-h627-no" width="600"></p>

### EMNIST By_Merge Dataset

* Best train accuracy: 82.18% with learning rate = 0.85 after 191,300 steps
* Best test accuracy: 80.13% with learning rate = 0.85 after 160,000 steps

<p align="center"><img src="https://lh3.googleusercontent.com/qcvhqxtIppFpG8JDq6A0ZqQ9jfgOvYU6vLy3cRSosaSjh8aoeO8_114_Bs1zzJOdhBHWBLa2XwYP4uKAhLbwvIZp--KxHe-T-wssmAnPmDAf0TMs6yYpaXaMk1McNcR9jM1JDLufrjuDNddcnCLQKXTv18Xe1vqtz_Dm2Cp0VTQ3tuV-tGvZh-7YSVJEG6u4dMMgfMV0HOUsstr-BQiM7ES2iUKhkMHo3Y-h4sGu5QbDu_ngM5eyPAwOBjpSY91vYBbSJC0eY_PMM7C3ZPDuD4EwscPyygck9tjXB2AHnq_Tu4eRxPrbt_gehYwQEKfx2ntt5Lm6X5lVjY45BKzpUInwRZ6yno3Vlly7L4Yf8vxNGi5Ob9A-PuwXhelnxA-Jk0BcTIF21hcu2ptrWzWEyhKoexaFyv8XDbydaMCWDHtw3TuVp_qYB7nan5CeHZBtBftBctUEBTnHWd2gBcnY4BWYeJn8-EB00tzxhn7ThIzVH4uy1GfilixFvBaFyXmFGhk8Lf8iHPvQeQIjhGG7ytYlN8soiMIdBQznP1nkgRQMC1ftKOdWo_oZW1GdC_5bloOirMuyqX1oiTXJOk-bp9RiWpCt3q7SNkqK9wc=w1319-h741-no" width="600"></p>


### EMNIST By_Class Dataset

* Best train accuracy: 87.00% with learning rate = 0.85 after 206,900 steps
* Best test accuracy: 85.73% with learning rate = 0.85 after 210,000 steps

<p align="center"><img src="https://lh3.googleusercontent.com/EkCTL8AVmSxlcJ66ypxL9ud7EUsYE-aFcaOdda9eOfUuOIEAwxV59xj22AX-tRNT5MWgk-oKOOTezkrhj-l4ZAPmbN-gyVf3_ZtjlePgZpJxyHISXD74pT0HMvuZEsOqySRO0AQzbj56k5dgrQTp5XuhJK_Mie1HJQBnooizTZJb-sZ0m4iYKhq3DwGW_cnTNkZQvekz9fNv2snrCNy41zvBXO4kRQC2TCiVmKTC0yLXwKLT9mFHwtSBKTN_v5LXXofeAKXHo0hDXr1msBf885etgKFQW4HXtR6K-0sMWwc5cUjAXSADi9fsZMrzvXRv-Z0mPxeNRgkETCltOUyhrKbCOza5RoY9YAz62wF-ZPq0tWNk0c0-Hg5ENXEfozFRMdSWAzzSh8wJF4-2VYAku5nRfVoTE_kx1RgezvCIpekYA0_nPKa0lJVVvXeljAUkoh_biROnuTHBACAUj_7-yqe4NNupw5UVvB1qa9bhLCxhytlqTehTyb3oLROsDVjmJkzEdzrrIP1l8OgF7jr2VB3PkVqn1ge0fK626nDCHf653tTaqlqYCCWfnxKd1QkwnN0bGiMGNgTcig32I6XnWqB-4jPB8AZVg9QE66M=w1275-h709-no" width="600"></p>

## AUTHOR
> **Bui The Viet** - *FPT University* - vietpro213tb@gmail.com
