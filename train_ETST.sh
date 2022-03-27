OUTPATH=experiments/test
WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/wiki/processed_data_cleaned/original/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/baidu/processed_data_cleaned/original/"
PRETRAINED=""
MODEL="/home/zchen/encyclopedia-text-style-transfer/cnn-text-classification-pytorch/experiments/cleaned/2022-03-17_14-49-32/best.pt"
CHECKPOINT=""

SENTENCE="台湾社会自1940年代以来开始出现要求成为完全之独立国家的呼声，其所代表的支持两岸统一的「统派」与支持台湾独立的「独派」并列为台湾当前的两大政治思想。"
# "分裂祖国的图谋是不会得逞的，必然遭到包括台湾同胞在内的全体中国人民的反对。"
# "台湾社会自1940年代以来开始出现要求成为完全之独立国家的呼声，其所代表的支持两岸统一的「统派」与支持台湾独立的「独派」并列为台湾当前的两大政治思想。"
# "台湾是中国自古以来不可分割的一部分。"
# "这 种 烧 制 方 法 的 优 点 , 是 最 大 限 度 地 利 用 空 位 空 间 , 既 可 节 省 燃 料 , 又 可 防 止 器 具 变 形 , 从 而 降 低 了 成 本 , 大 幅 度 地 提 高 了 产 量 , 对 南 北 瓷 窑 都 产 生 过 很 大 影 响 , 对 促 进 我 国 制 瓷 业 的 发 展 起 了 重 要 作 用 ."
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
    -snapshot $MODEL \
    \
    `# for prediction \
    # -test \
    # -predict_probs` \
    -predict "$SENTENCE" \
    # -not_tokenize