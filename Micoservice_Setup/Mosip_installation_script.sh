#!/bin/bash

## Steps to do manually before launching this script:
# 1. Login to the console machine [done]
# 2. Create this file on the machine and call it script.sh [done]
# 3. Run chmod +x script.sh [done]
# 4. Add your private and public key that you are using for CloudLab (preferably passwordless) to .ssh and rename them tempkey and tempkey.pub
# 5. Reset permissions of the key by running sudo chmod 600 tempkey and sudo chmod tempkey.pub
# 6. Replace USERNAME by the user associated with your key
# 7. Replace MZWORKER0 and DMZWORKER0 by the IP address of mzworker0 and dmzworker0 respectively (you can find that information by pinging them)
# 8. Go to https://www.google.com/recaptcha/admin/create and create a captcha (captcha v2 im not a robot). In the domain section, enter https:// followed by the hostname of console
# 9. Replace CAPTCHASITE and CAPTCHASECRET with the captcha site and secret keys generated

USERNAME='ENTER_YOUR_USERNAME'
MZWORKER0='ENTER_YOUR_MZWORKER0_IP_ADDRESS'
DMZWORKER0='ENTER_YOUR_DMZWORKER0_IP_ADDRESS'
CAPTCHASITE='ENTER_YOUR_CAPTCHA_SITE_KEY'
CAPTCHASECRET='ENTER_YOUR_CAPTCHA_SECRET_KEY'

HOSTS='mzmaster mzworker0 mzworker1 mzworker2 mzworker3 mzworker4 mzworker5 mzworker6 mzworker7 mzworker8 dmzmaster dmzworker0'
HOSTNAME=$(hostname)

## Adding user mosipuser and generating his ssh keys
sudo useradd mosipuser
echo -e "mosippwd\nmosippwd" | sudo passwd mosipuser
sudo usermod -aG wheel mosipuser
echo "mosipuser ALL=(ALL)  ALL" | sudo EDITOR='tee -a' visudo
echo "%mosipuser  ALL=(ALL)  NOPASSWD:ALL" | sudo EDITOR='tee -a' visudo

sudo -i -u mosipuser bash << EOF
mkdir ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t rsa -N "" -f .ssh/id_rsa
EOF

sudo cp /users/${USERNAME}/.ssh/tempkey /home/mosipuser/.ssh/tempkey && sudo chown mosipuser:mosipuser /home/mosipuser/.ssh/tempkey
sudo cp /users/${USERNAME}/.ssh/tempkey.pub /home/mosipuser/.ssh/tempkey.pub && sudo chown mosipuser:mosipuser /home/mosipuser/.ssh/tempkey.pub
sudo cp /home/mosipuser/.ssh/id_rsa.pub /users/${USERNAME}/.ssh/id_rsa.pub

KEY=$(<.ssh/id_rsa.pub)

sudo -i -u mosipuser bash << EOF2
sudo echo -e "$KEY" >> .ssh/authorized_keys
chmod 644 .ssh/authorized_keys
EOF2

## Adding his ssh keys to the authorized list of the other hosts
SCRIPT1="echo '"
SCRIPT2="' | sudo tee -a /root/.ssh/authorized_keys && sudo systemctl stop firewalld && sudo systemctl disable firewalld"
SCRIPT="$SCRIPT1$KEY$SCRIPT2"

for HOST in ${HOSTS} ; do
    ssh -tt -i .ssh/tempkey -o "StrictHostKeyChecking no" -l ${USERNAME} ${HOST} "${SCRIPT}"
done

sudo -i -u mosipuser bash << EOF3
## Downloading MOSIP and other components
sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum install -y ansible
sudo yum install -y git
git clone https://github.com/mosip/mosip-infra.git
sudo chown -R mosipuser mosip-infra/

## Changing MOSIP naming convention
sudo sed -i "s/.*cluster_master:.*/cluster_master: 'mzmaster'/" mosip-infra/deployment/sandbox-v2/group_vars/mzcluster.yml
sudo sed -i "s/.*cluster_master:.*/cluster_master: 'dmzmaster'/" mosip-infra/deployment/sandbox-v2/group_vars/dmzcluster.yml

sudo sed -i 's/.*nodeport_node: mzworker0.sb.*/    nodeport_node: mzworker0/' mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i 's/.*nodeport_node: dmzworker0.sb.*/    nodeport_node: dmzworker0/' mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*host: 'mzworker0.sb:30080\/elasticsearch'.*/        host: 'mzworker0:30080\/elasticsearch'/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*url: http:\/\/mzworker0.sb:30601.*/        url: http:\/\/mzworker0:30601/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*node: 'mzworker0.sb'.*/    node: 'mzworker0'/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*server: console.sb.*/  server: console/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*node: 'mzworker1.sb'.*/    node: 'mzworker1'/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml

sudo sed -i "s/console.sb ansible_user=mosipuser/console ansible_user=mosipuser/g" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*dmzmaster.sb ansible_user=root.*/dmzmaster ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*dmzworker0.sb ansible_user=root.*/dmzworker0 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzmaster.sb ansible_user=root.*/mzmaster ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker0.sb ansible_user=root.*/mzworker0 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker1.sb ansible_user=root.*/mzworker1 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker2.sb ansible_user=root.*/mzworker2 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker3.sb ansible_user=root.*/mzworker3 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker4.sb ansible_user=root.*/mzworker4 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker5.sb ansible_user=root.*/mzworker5 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker6.sb ansible_user=root.*/mzworker6 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker7.sb ansible_user=root.*/mzworker7 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini
sudo sed -i "s/.*mzworker8.sb ansible_user=root.*/mzworker8 ansible_user=root/" mosip-infra/deployment/sandbox-v2/hosts.ini

sudo sed -i "s/.*master: dmzmaster.sb.*/    master: dmzmaster/" mosip-infra/deployment/sandbox-v2/playbooks/dmzcluster.yml
sudo sed -i "s/.*master: mzmaster.sb.*/    master: mzmaster/" mosip-infra/deployment/sandbox-v2/playbooks/mzcluster.yml

## Changing MOSIP settings
sudo sed -i "s/.*sandbox_domain_name:.*/sandbox_domain_name: $HOSTNAME/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i 's/.*network_interface:.*/network_interface: "eth1"/' mosip-infra/deployment/sandbox-v2/group_vars/mzcluster.yml
sudo sed -i 's/.*network_interface:.*/network_interface: "eth1"/' mosip-infra/deployment/sandbox-v2/group_vars/dmzcluster.yml

## Add nfsnobody as it fails without it
sudo useradd nfsnobody

## Add shortcuts
sudo echo -e "alias an='ansible-playbook -i hosts.ini'" >> ~/.bashrc
sudo echo -e "alias kc1='kubectl --kubeconfig \$HOME/.kube/mzcluster.config'" >> ~/.bashrc
sudo echo -e "alias kc2='kubectl --kubeconfig \$HOME/.kube/dmzcluster.config'" >> ~/.bashrc
sudo echo -e "alias sb='cd \$HOME/mosip-infra/deployment/sandbox-v2/'" >> ~/.bashrc
sudo echo -e "alias helm1='helm --kubeconfig \$HOME/.kube/mzcluster.config'" >> ~/.bashrc
sudo echo -e "alias helm2='helm --kubeconfig \$HOME/.kube/dmzcluster.config'" >> ~/.bashrc
source ~/.bashrc

## Setting Proxy OTP mode
sudo sed -i 's/.*mosip.kernel.sms.proxy-sms.*/mosip.kernel.sms.proxy-sms=true/' mosip-infra/deployment/sandbox-v2/roles/config-repo/files/properties/application-mz.properties
sudo sed -i 's/.*mosip.kernel.auth.proxy-otp.*/mosip.kernel.auth.proxy-otp=true/' mosip-infra/deployment/sandbox-v2/roles/config-repo/files/properties/application-mz.properties
sudo sed -i 's/.*mosip.kernel.auth.proxy-email.*/mosip.kernel.auth.proxy-email=true/' mosip-infra/deployment/sandbox-v2/roles/config-repo/files/properties/application-mz.properties

## Changing mzworker0 and dmzworker0 IP addresses
sudo sed -i "s/.*any_node_ip: '10.20.20.157'.*/    any_node_ip: '$MZWORKER0'/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml
sudo sed -i "s/.*any_node_ip: '10.20.20.207'.*/    any_node_ip: '$DMZWORKER0'/" mosip-infra/deployment/sandbox-v2/group_vars/all.yml

## Adding Captcha
sudo sed -i "s/.*google.recaptcha.site.key=.*/google.recaptcha.site.key=$CAPTCHASITE/" mosip-infra/deployment/sandbox-v2/roles/config-repo/files/properties/pre-registration-mz.properties
sudo sed -i "s/.*google.recaptcha.secret.key=.*/google.recaptcha.secret.key=$CAPTCHASECRET/" mosip-infra/deployment/sandbox-v2/roles/config-repo/files/properties/pre-registration-mz.properties

## Launching ansible MOSIP installation script
sb
an site.yml
EOF3
