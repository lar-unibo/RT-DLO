# RT-DLO: Real-Time Deformable Linear Objects Instance Segmentation

Deformable linear objects (DLOs), such as cables, wires, ropes, and elastic tubes, are numerously present both in domestic and industrial environments.
Unfortunately, robotic systems handling DLOs are rare and have limited capabilities due to the challenging nature of perceiving them. Hence, we propose a novel approach named RT-DLO for real-time instance segmentation of DLOs. 

First, the DLOs are semantically segmented from the background. Afterward, a novel method to separate the DLO instances is applied. 

It employs the generation of a graph representation of the scene givencthe semantic mask where the graph nodes are sampled from the DLOs center-lines whereas the graph edgescare selected based on topological reasoning. RT-DLO is experimentally evaluated against both DLO-specific and general-purpose instance segmentation deep learning approaches, achieving overall better performances in terms of accuracy and inference time

<div align="center">
<h2> RT-DLO: Real-Time Deformable Linear Objects Instance Segmentation </h2>

 :page_with_curl:  [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10045806)  :page_with_curl:	
</div>

### Setting up python environment

```
conda env create -f env.yml

conda activate rt-dlo

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

conda install pyg -c pyg
```

### Download checkpoints and test data

checkpoints and data available [here](https://mega.nz/file/gMNkGZoa#wYIoMfsRpV_vF5yIp_udVtj6iRrErmQGv2VCiUQgY-M).


### Citation
If you our research interesting, please cite the following manuscript.
```
@ARTICLE{10045806,
  author={Caporali, Alessio and Galassi, Kevin and Žagar, Bare Luka and Zanella, Riccardo and Palli, Gianluca and Knoll, Alois C},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={RT-DLO: Real-Time Deformable Linear Objects Instance Segmentation}, 
  year={2023},
  volume={19},
  number={11},
  pages={11333-11342},
  doi={10.1109/TII.2023.3245641}}

```
