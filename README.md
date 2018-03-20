# tpt39-athens

Semaine Athens 19-24 March, 2018

# Getting started

Add the following to your ssh config (`nano ~/.ssh/config`)

```
Host tpt_bastion
  Hostname ssh2.enst.fr
  User <YOUR_USERNAME>
  ForwardAgent yes
  IdentityFile ~/.ssh/id_rsa

Host odroid
  Hostname a405-15-arm.enst.fr
  ProxyCommand ssh tpt_bastion -W %h:%p
  User odroid
  IdentityFile ~/.ssh/id_rsa
  ForwardAgent yes

Host *.enst.fr
  ProxyCommand ssh tpt_bastion -W %h:%p
  User <YOUR_USERNAME>
  IdentityFile ~/.ssh/id_rsa
  ForwardAgent yes
```

If you don't have an ssh key (check if you have the file `~/.ssh/id_rsa`), generate one with

```
ssh-keygen
```

Then copy your ssh key to both the tpt_bastion and the odroid host with

```
ssh-copy-id tpt_bastion && ssh-copy-id odroid
```
