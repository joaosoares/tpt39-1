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
