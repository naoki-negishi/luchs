# luchs

## 実行方法
```docker compose -f Docker/docker-compose.yml up -d --build```
```docker container run -itd --rm --name hol --gpus '"device=0,1"' -v /work01/negishi_naoki/syntactic/luchs:/code/ docker_nli```

## 訓練 (コンテナ内で)
```python training/training.py --yaml_file config/test.yaml```
