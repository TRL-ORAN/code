# Autoregressive Model for 5G O-RAN: A Transformer-Based Reinforcement Learning Framework for Joint Handover and Scheduling Optimization
User dynamics constitute a defining characteristic of cellular networks. However, existing reinforcement
learning (RL)-based network optimization approaches largely overlook this critical aspect. Most of them
implicitly assume that a fixed number of user equipment (UE) devices participate in communication, fundamentally limiting their adaptability in practical deployments. 

In this paper, we present a Transformer-based Reinforcement Learning (TRL) framework to address user dynamics for joint handover and scheduling optimization in 5G Open Radio Access Networks (O-RANs), with the objective of maximizing a Quality of Service (QoS)-based reward function. TRL uniquely treats each UE’s measurement data as an input token and utilizes the Transformer architecture’s attention mechanisms to generate context-aware, per-UE policy decisions sequentially. Crucially, TRL naturally adapts to the networks with a time-varying number of UEs by employing dynamic causal masking, thus eliminating the rigid, fixed input sizes or zero-padding strategies required by previous methods. 

We implemented and validated the TRL framework on a real-world 5G O-RAN testbed comprising three cells and 15 smartphones operating under realistic traffic conditions. Extensive over-the-air experiments demonstrate the practicality of TRL and its superior performance compared to state-of-the-art
RL-based methods in dynamic environments. Source code is available at: https://github.com/TRL-ORAN/code.

## Getting Started
### Minimum hardware requirements:
- Laptop/Desktop/Server for OAI CN5G and OAI gNB
    - Operating System: Ubuntu 22.04 LTS
    - CPU: 12 cores x86_64 @ 3.5 GHz
    - RAM: 32 GB
<!-- - [Commercial RU ORAN_FHI7.2](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/ORAN_FHI7.2_Tutorial.md) -->
The hardware on which we used with commercial RUs:

|Hardware (CPU,RAM)                       |Operating System (kernel)                       |NIC (Vendor,Driver,Firmware)             |
|-----------------------------------------|------------------------------------------------|-----------------------------------------|
|Intel(R) Core(TM) i9-14900K 24-Core, 32GB|Ubuntu 22.04.5 LTS (5.15.0-1033-realtime)       |Intel Ethernet (E800 Series), ice (2.2.9), 4.80 0x80020682 1.3805.0 |


### Software reference:
Our system consists of the configuration of RIC + xApp, 5G O-RAN, and 5G Core. TRL has developed its own architecture and algorithms based on [Flexric](https://gitlab.eurecom.fr/mosaic5g/flexric), [OAI cn5g](https://gitlab.eurecom.fr/oai/cn5g), and [openairinterface5G](https://gitlab.eurecom.fr/oai/openairinterface5g).

## Dependencies and Code Install

## 1. RIC + xApp Setup (RIC)

### 1.1 python conv requirement
1. First, ensure that Python and pip are installed on the target computer. You can check this by running the following commands:
```bash
python --version 
pip --version
```
If Python is not installed, please download and install it from the official Python website.

2. In the target directory, create and activate a new virtual environment and install dependencies. 
```bash
python -m venv myenv  
source myenv/bin/activate
```
Copy the requirements.txt file to your target computer's directory. Then, install all the dependencies using the following command:
```bash
conda env create -f environment.yml
```


### 1.2 RIC prerequisites

Please find the CMAKE and SWIG dependencies on the [Flexric website](https://gitlab.eurecom.fr/mosaic5g/flexric).

Note: - GCC (gcc-10)      *gcc-11 is not currently supported.

### 1.3 Build RIC +xApp code and install it. 

* Make install
```bash
# Build RIC
cd RIC && mkdir build && cd build && cmake .. && make -j8 
# You can install the Service Models (SM) in your computer via:
sudo make install
```

## 2. O-RAN/gNB Setup  (GNB)

### 2.1 5G Core Setup

Please install and configure OAI CN5G as described here: [OAI 5G NR CN tutorial](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_OAI_CN5G.md)

### 2.2 Pre-requisites

Please find the UHD and other requirements on the [OAI gNB tutorial](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/NR_SA_Tutorial_COTS_UE.md).

 <!-- Build UHD from source
```bash
sudo apt install -y autoconf automake build-essential ccache cmake cpufrequtils doxygen ethtool g++ git inetutils-tools libboost-all-dev libncurses-dev libusb-1.0-0 libusb-1.0-0-dev libusb-dev python3-dev python3-mako python3-numpy python3-requests python3-scipy python3-setuptools python3-ruamel.yaml

git clone https://github.com/EttusResearch/uhd.git ~/uhd
cd ~/uhd
git checkout v4.7.0.0
cd host
mkdir build
cd build
cmake ../
make -j $(nproc)
make test # This step is optional
sudo make install
sudo ldconfig
sudo uhd_images_downloader
``` -->
### 2.3 Install and Build GNB

Please find how to set up commercial RUs with O-DU on [ORAN_FHI7.2 website](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/ORAN_FHI7.2_Tutorial.md)

- Smartphone set up
The COTS UE can now search for the network. You can find how to connect UE to gNB on [srsRAN website](https://docs.srsran.com/projects/project/en/latest/tutorials/source/cotsUE/source/index.html).

### 2.4 Separate O-CU and  O-DU

Please find how to separate O-CU and  O-DU on [F1AP website](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/F1AP/F1-design.md?ref_type=heads).


## Modified code structure
In both the RAN and RIC systems, there are numerous code files involved. Below, I have listed the files that I modified or added as part of implementing TRL. The structure is as follows. For detailed comments and further infortrlion, please refer directly to the code.

### 1. TRL_xApp

<!-- # ├── du-data                             # data storage
# │   ├── KPM_UE.txt             
# │   ├── ctrl.csv                     # sent scheduling decision to CU and each DU         
# │   ├── xapp_db_               
# │   ├── kpm.py                       # show KPM data
# │   └── rewards.csv          -->

```bashs
├── gnb                                            # source code for gnb
    ├── openair2                     
    │   ├── E2AP                                   # source code for E2 interface
       │   ├── RAN_FUNCTION                   
           │   ├── CUSTOMIZED                      # monitor functions
               │   ├── ran_func_mac.c         
               │   └── ran_func_kpm.c         
           │   └── O-RAN                           # control service functions
               │   ├── ran_func_rc.h               # get interfarence map
               │   ├── rc_ctrl_service_style_2.h   
       ├── LAYER2                                  # MAC layer funtions
       │   ├── NR_MAC_gNB                          # MAC scheduler      
          │   ├── gNB_scheduler_dlsch.c            # source code for downlink mac scheduler
├── examples                                       
│   ├── ric                                        # RIC including E2 interface
│   └── xApp                                       # source code for xApps
    │   c                                          # our xApps based on C code
        │   ctrl            
        │   ├── mac_ctrl.c                         # test code for mac layer control
        │   ├── xapp_handover_TRL.c                # xApp for TRL
├── TRL                                            
│   └── train_xapp.sh                              # our TRL algrithm


```

### 2. Extend TRL
If you wish to extend 'TRL', please review the modification sections and comments in the code above. These will guide you through quickly getting started with implementing your own online learning algorithm in a new xApp.

## Run TRL
## 1. On 5G Core server:
```bash
cd ~/oai-cn5g
docker compose down
docker compose up -d
```
## 2. On gNB server:

### 2.1 Run O-CU
```bash
cd ~/openairinterface5g
sudo cmake_targets/ran_build/build/nr-softmodem --sa -O gnb-cu.sa.band78.106prb.conf --telnetsrv --telnetsrv.shrmod ci --telnetsrv.listenaddr 127.0.0.1 --telnetsrv.listenport 2001
```

### 2.1 Run 3 O-DU-O-RU (Cells): see [ORAN_FHI7.2 website](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/doc/ORAN_FHI7.2_Tutorial.md)


### 2.2 Check UEs' successfully connected and generate demand:
You can use the following commands to check the 5G Core's AMF, UPF, and other components.
```bash
docker logs oai-amf -f
docker logs oai-upf -f
```

You can generate traffic by accessing websites, streaming videos, downloading content, and more through the UE. Additionally, you can create traffic demands using iperf.

Run Iperf on UE to generate demand:
```bash
docker exec -it oai-ext-dn bash
iperf -u -t 86400 -i 1 -fk -B 192.168.70.135 -b 10M -c 10.0.0.2
```

### 2.3 Additional infortrlion to the manual HO trigger
You can trigger the handover manually by logging in through telnet:
```bash
telnet 127.0.0.1 2001
ci trigger_f1_ho 1
ci trigger_f1_ho_to_du 1 19
ci trigger_ho_to_cell 1 3587    
```

## 3. On RIC+xApp server:
The RIC is deployed on an edge server with four NVIDIA A6000 GPUs.
### 3.1 Start the nearRT-RIC
```bash
./RIC/build/examples/ric/nearRT-RIC
```

### 3.2 Start xApps

Start the TRL xApp

```bash
./RIC/build/examples/xApp/c/ctrl/xapp_handover_TRL
 ```
### 3.3 Start testing our algrithom
* To evaluate TRL in your host:
```bash
cd ~/TRL
./train_xapp.sh
```

## Citation
If you use our code in your research, please cite our paper:
```bash
Coming soon...
```

## Getting help

If you encounter a bug or have any questions regarding the paper, the code or the setup process, please feel free to contact us: Coming soon...
