OUTPATH=experiments/2
WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/data/wiki/processed_data/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/data/baidu/processed_data/tokenized/"
PRETRAINED=""
MODEL="/home/zchen/encyclopedia-text-style-transfer/cnn-text-classification-pytorch/experiments/2/2022-01-03_12-31-57/best.pt"
CHECKPOINT=""

SENTENCE="分裂祖国的图谋是不会得逞的，必然遭到包括台湾同胞在内的全体中国人民的反对。"
# "台湾社会自1940年代以来开始出现要求成为完全之独立国家的呼声，其所代表的支持两岸统一的「统派」与支持台湾独立的「独派」并列为台湾当前的两大政治思想。"
# "台湾是中国不可分割的一部分。"
# "隋 吉 藏 在 此 创 立 中 国 佛 教 三 论 宗 .   为 中 国 国 内 佛 教 四 大 丛 林 之 一 ."
# "美 国 方 面 声 明 : 美 国 认 识 到 , 在 台 湾 海 峡 两 边 的 所 有 中 国 人 都 认 为 只 有 一 个 中 国 , 台 湾 是 中 国 的 一 部 分 .   美 国 政 府 对 这 一 立 场 不 提 出 异 议 .   它 重 申 它 对 由 中 国 人 自 己 和 平 解 决 台 湾 问 题 的 关 心 .   考 虑 到 这 一 前 景 , 它 确 认 从 台 湾 撤 出 全 部 美 国 武 装 力 量 和 军 事 设 施 的 最 终 目 标 .   在 此 期 间 , 它 将 随 着 这 个 地 区 紧 张 局 势 的 缓 和 逐 步 减 少 它 在 台 湾 的 武 装 力 量 和 军 事 设 施 ."

export CUDA_VISIBLE_DEVICES=0

python main.py \
    -log-interval 100 \
    -test-interval 6000 \
    -save-interval 6000 \
    -save-dir $OUTPATH \
    -early-stop 20 \
    \
    -dataset ETST \
    -wiki_dir $WIKIPATH \
    -baidu_dir $BAIDUPATH \
    \
    `# for prediction` \
    -snapshot $MODEL \
    -predict "$SENTENCE" \
    -tokenize true