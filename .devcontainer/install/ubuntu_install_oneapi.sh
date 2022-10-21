apt-get install -y --no-install-recommends curl ca-certificates gpg-agent software-properties-common

# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

apt update && apt install intel-basekit intel-hpckit -y

echo "source /opt/intel/oneapi/compiler/latest/env/vars.sh" >> ~/.bashrc
echo "source /opt/intel/oneapi/vtune/latest/vtune-vars.sh" >> ~/.bashrc




