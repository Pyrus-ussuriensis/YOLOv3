def parse_cfg(cfg_path):
    blocks = []
    with open(cfg_path, 'r') as f:
        block = None
        for line in f:
            line = line.strip()
            # 忽略空行和注释
            if not line or line.startswith('#'):
                continue
            # 新区块开始
            if line.startswith('[') and line.endswith(']'):
                if block:
                    blocks.append(block)
                block = {'type': line[1:-1]}
            else:
                # 普通键值对
                key, val = line.split('=', 1)
                block[key.strip()] = val.strip()
        if block:
            blocks.append(block)
    return blocks

# 使用示例
cfg_blocks = parse_cfg('models/yolov3.cfg')
for i, blk in enumerate(cfg_blocks):
    print(f"Layer {i}: {blk['type']} with params { {k:blk[k] for k in blk if k!='type'} }")
