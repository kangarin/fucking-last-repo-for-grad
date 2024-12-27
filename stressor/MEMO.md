
## cpu压测
stress-ng直接调用就行。

## gpu压测

没什么现成的工具，可能得自己写了。先得在jetson设备上装pycuda。可以参考这个脚本，做点改动。

https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/install_pycuda.sh

下面是我在tx2上实际能成功安装pycuda的脚本。

```
#!/bin/bash

set -e

export PATH=/usr/local/cuda/bin:$PATH
if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

arch=$(uname -m)
folder=${HOME}/src
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y build-essential python3-dev python3-pip
sudo apt-get install -y libboost-python-dev libboost-thread-dev
sudo pip3 install setuptools

# 直接指定 Python3 的 boost 库名
boost_pyname=boost_python3-py36

echo "** Download pycuda-2019.1.2 sources"
pushd $folder
if [ ! -f pycuda-2019.1.2.tar.gz ]; then
  wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
fi

echo "** Build and install pycuda-2019.1.2"
CPU_CORES=$(nproc)
echo "** cpu cores available: " $CPU_CORES
tar xzvf pycuda-2019.1.2.tar.gz
cd pycuda-2019.1.2
python3 ./configure.py --python-exe=/usr/bin/python3 --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib/${arch}-linux-gnu --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib/${arch}-linux-gnu --boost-python-libname=${boost_pyname} --boost-thread-libname=boost_thread --no-use-shipped-boost
make -j$CPU_CORES
python3 setup.py build
sudo python3 setup.py install

popd

python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"
```