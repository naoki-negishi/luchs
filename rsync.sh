#!/bin/zsh

# from local to inui server
rsync -avu --progress \
      -e 'ssh -p 2022' \
      --exclude '.git' \
      --exclude '.DS_Store' \
      --exclude '.gitignore' \
      --exclude 'rsync.sh' \
      $HOME/Projects/syntactic/luchs/ \
      negishi_naoki@cocoa.nlp.ecei.tohoku.ac.jp:/home/negishi_naoki/syntactic/luchs

echo 'rsync local:~/Projects/syntactic -> cocoa:/home/negishi_naoki/syntactic'

ssh sherry01 rsync -avu --progress \
      --exclude '.DS_Store' \
      /home/negishi_naoki/syntactic/luchs/ \
      /work_share/negishi_naoki/syntactic/luchs

echo 'rsync cocoa:/home/negishi_naoki/syntactic -> sherry01:/work01/negishi_naoki/syntactic'

echo '============================================================================================='

# from inui server to local
ssh sherry01 rsync -avu --progress \
      --exclude '.DS_Store' \
      --exclude 'wandb' \
      /work_share/negishi_naoki/syntactic/luchs/ \
      /home/negishi_naoki/syntactic/luchs

echo 'rsync cocoa:/home/negishi_naoki/syntactic -> sherry01:/work01/negishi_naoki/syntactic'

rsync -avhu --progress \
      -e 'ssh -p 2022' \
      --exclude '.DS_Store' \
      --exclude 'wandb' \
      negishi_naoki@cocoa.nlp.ecei.tohoku.ac.jp:/home/negishi_naoki/syntactic/luchs/ \
      $HOME/Projects/syntactic/luchs

echo 'rsync sherry01:/home/negishi_naoki/syntactic -> local:~/Projects'
