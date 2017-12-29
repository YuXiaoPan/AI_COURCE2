package me.peyppicp.cnn;

import com.vdurmont.emoji.EmojiParser;
import lombok.Data;
import me.peyppicp.Utils;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Data
public class EmojiLimiter {

    public static final String UNKNOWN = "$UNKNOWN";
    public static final String STOP = "$STOP";

    private String samplesFilePath;
    private String wordVectorPath;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private File samplesFile;

    private Map<String, Integer> wordIndexMap;
    private int outComesNum;
    private final int limitNum;

    public EmojiLimiter(String samplesFilePath, int limitNum) throws IOException {
        this.samplesFilePath = samplesFilePath;
        this.limitNum = limitNum;
        init();
    }

    public void writeToFile() {
        List<String> temp = new ArrayList<>();
        wordIndexMap.forEach((s, integer) -> temp.add(s + "," + integer));
        try {
            Utils.writeLineToPath(temp, CNNDecideEmojiMain.PREFIX + "pair_for_emoji.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void init() throws IOException {
        samplesFile = new File(samplesFilePath);
        wordIndexMap = new HashMap<>();
        List<String> samples = FileUtils.readLines(samplesFile, Charsets.UTF_8);
        for (String sample : samples) {
            List<String> extractEmojis = EmojiParser.extractEmojis(sample)
                    .parallelStream().distinct().collect(Collectors.toList());
            for (String extractEmoji : extractEmojis) {
                wordIndexMap.merge(extractEmoji, 1, (o, n) -> o + n);
            }
        }

//        按照value排序
        Map<String, Integer> temp = new LinkedHashMap<>();
        wordIndexMap.entrySet()
                .parallelStream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(limitNum)
                .forEachOrdered(entry -> temp.put(entry.getKey(), entry.getValue()));

        wordIndexMap = new LinkedHashMap<>();
        int index = 0;
        for (String emoji : temp.keySet()) {
            wordIndexMap.putIfAbsent(emoji, index++);
        }
//        addUnknown();
        this.outComesNum = wordIndexMap.keySet().size();
    }

    public void addUnknown() {
        int size = wordIndexMap.values().size();
        wordIndexMap.putIfAbsent(UNKNOWN, size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
    }
}
