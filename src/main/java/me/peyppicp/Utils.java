package me.peyppicp;

import com.google.common.base.Preconditions;
import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.cnn.EmojiLimiter;
import me.peyppicp.cnn.EmojiToIndex;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/16
 * @email yuxiao.pan@kikatech.com
 */
public class Utils {

    public static void writeLineToPath(List<String> data, String path) throws IOException {
        FileUtils.writeLines(new File(path),
                "UTF-8",
                data,
                "\n",
                false);
    }

    public static List<String> readLinesFromPath(String path) throws IOException {
        return FileUtils.readLines(new File(path), Charsets.UTF_8);
    }

    /**
     * CNNMain处理emoji_sample.txt文件
     * 过滤只包含emoji表情的行
     *
     * @param input     输入数据源地址
     * @param output    输出数据源地址
     * @param addOrigin 是否保留原有行
     * @throws IOException
     */
    public static void processOriginalSamples(String input, String output, boolean addOrigin) throws IOException {
        List<String> sampleLines = FileUtils.readLines(new File(input), Charsets.UTF_8);
        List<String> emojiUnicodes = EmojiManager.getAll().parallelStream().map(Emoji::getUnicode).collect(Collectors.toList());
        List<String> temp = new ArrayList<>();
        List<String> errorLines = new ArrayList<>();
        int count = 0;
        int totalSize = sampleLines.size() / 1000;

//        sampleLines = sampleLines.parallelStream().map(EmojiParser::parseToUnicode).collect(Collectors.toList());

//        按照emoji进行切分
        for (String line : sampleLines) {
            try {
                int emojiLength = 2;
                int currentEmojiIndex = 0;
                List<String> containedEmojis = EmojiParser.extractEmojis(line).parallelStream().distinct().collect(Collectors.toList());
                if (!containedEmojis.isEmpty()) {
                    for (String emoji : containedEmojis) {
                        boolean flag = false;
                        currentEmojiIndex = line.indexOf(emoji);
                        if (currentEmojiIndex != -1) {
                            for (int i = currentEmojiIndex; i < line.length() - 1; i += emojiLength) {
                                if (EmojiManager.isEmoji(line.substring(currentEmojiIndex, currentEmojiIndex + emojiLength))) {
                                    currentEmojiIndex += emojiLength;
                                    flag = true;
                                }
                            }
                        }
                        if (flag) {
                            temp.add(line.substring(0, currentEmojiIndex).trim().toLowerCase());
                            line = line.substring(currentEmojiIndex, line.length()).trim().toLowerCase();
                        }
                    }
                } else {
                    if (addOrigin) {
                        temp.add(line);
                    }
                }
            } catch (Exception e) {
                errorLines.add(line);
            }
            count++;
            if (count % 1000 == 0) {
                System.out.println("Remain: " + (totalSize - (count / 1000)));
            }
        }

        temp = temp.parallelStream().filter(s -> EmojiParser.extractEmojis(s).size() != s.length() / 2).distinct().collect(Collectors.toList());
        List<String> temp1 = new ArrayList<>();

//        添加空格
        for (String sample : temp) {
            List<String> extractEmojis = EmojiParser.extractEmojis(sample);
            if (!extractEmojis.isEmpty()) {
                String emoji = extractEmojis.get(0);
                int i = sample.indexOf(emoji);
                if (i >= 1) {
                    if (i == emoji.length() - 1) {
                        continue;
                    } else {
                        String head = sample.substring(0, i);
                        String last = sample.substring(i, sample.length());
                        temp1.add(head + " " + last);
                    }
                }
            } else {
                temp1.add(sample);
            }
        }

        List<String> temp2 = new ArrayList<>();
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        for (String sample : temp1) {
            List<String> tokens = tokenizerFactory.create(sample).getTokens();
            StringBuilder stringBuilder = new StringBuilder();
            tokens.forEach(s -> stringBuilder.append(s).append(" "));
            temp2.add(stringBuilder.toString().trim());
        }

        FileUtils.writeLines(new File(output),
                "UTF-8",
                temp2,
                "\n",
                false);
    }

    public static void markLabels(String input, String output) throws IOException {
        File file = new File(input);
//        File file = new File(PREFIX + "EmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        EmojiLimiter emojiLimiter = new EmojiLimiter(input, 25);
        emojiLimiter.writeToFile();
        EmojiToIndex emojiToIndex = new EmojiToIndex();
        ArrayList<String> labels = new ArrayList<>();
        for (String sample : samples) {
            List<String> emojis = EmojiParser.extractEmojis(sample)
                    .parallelStream().distinct().collect(Collectors.toList());
            StringBuilder sb = new StringBuilder();
            if (emojis.size() == 0) {
//                int index = EmojiToIndex.getIndex(EmojiToIndex.UNKNOWN);
                int index = -1;
                labels.add(String.valueOf(index));
                continue;
            }
            for (String emoji : emojis) {
                int index = emojiToIndex.getIndex(emoji);
                sb.append(index).append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            labels.add(sb.toString());
        }
        Preconditions.checkArgument(samples.size() == labels.size());
        FileUtils.writeLines(new File(output),
                "UTF-8",
                labels,
                "\n",
                false);
    }

    public static void removeEmojis(String input, String output) throws IOException {
        File file = new File(input);
//        File file = new File(PREFIX + "EmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        ArrayList<String> result = new ArrayList<>();
        for (String sample : samples) {
            String s = EmojiParser.removeAllEmojis(sample);
            result.add(s.trim());
        }

        Preconditions.checkArgument(samples.size() == result.size());
        FileUtils.writeLines(new File(output),
                "UTF-8",
                result,
                "\n",
                false);
    }
}
