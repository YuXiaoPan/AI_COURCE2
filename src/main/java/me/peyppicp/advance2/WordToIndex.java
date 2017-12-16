package me.peyppicp.advance2;

import com.vdurmont.emoji.EmojiParser;
import lombok.Data;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Data
public class WordToIndex {

    public static final String UNKNOWN = "$UNKNOWN";
    public static final String STOP = "$STOP";

    private String samplesFilePath;
    private String wordVectorPath;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private File samplesFile;

    private Map<String, Integer> wordIndexMap;
    private int outComesNum;

    public WordToIndex(String samplesFilePath) throws IOException {
        this.samplesFilePath = samplesFilePath;
        init();
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
                .limit(25)
                .forEachOrdered(entry -> temp.put(entry.getKey(), entry.getValue()));

        wordIndexMap = new LinkedHashMap<>();
        int index = 0;
        for (String emoji : temp.keySet()) {
            wordIndexMap.putIfAbsent(emoji, index++);
        }
//        addUnknown();
        this.outComesNum = wordIndexMap.keySet().size();
    }

    public int getIndex(String word) {
//        return wordIndexMap.getOrDefault(word, wordIndexMap.get(UNKNOWN));
        return wordIndexMap.getOrDefault(word, -1);
    }

    public void addUnknown() {
        int size = wordIndexMap.values().size();
        wordIndexMap.putIfAbsent(UNKNOWN, size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
    }

    public String getEmoji(int index) {
        return wordIndexMap.entrySet().parallelStream().filter(entry -> entry.getValue() == index).findFirst().get().getKey();
    }

    public int totalIndexNum() {
        return wordIndexMap.values().size();
    }

    public List<String> totalLabels() {
        return wordIndexMap.keySet().parallelStream().collect(Collectors.toList());
    }
}
