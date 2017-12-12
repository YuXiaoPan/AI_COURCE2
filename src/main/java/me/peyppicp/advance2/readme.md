# 调用顺序

- 执行ReformatEmojiSample2，生成EmojiSample.txt文件（只存在包含emoji的语句且emoji总是位于句未）
- 执行Word2VecMain，生成对应的word2vec。
- 执行BuildSampleLabel，构建EmojiSample.txt对应的emoji label。
- 执行RemoveEmojis，去除EmojiSample.txt中的语句中的emoji，至此，定型数据构建完成。