package me.peyppicp.cnn;

import me.peyppicp.Utils;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.IOException;
import java.util.*;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class WordLimiter {

    private List<String> lines;
    private Map<String, Integer> wordCounter;
    private TokenizerFactory tokenizerFactory;
    private int totalWordsCount;
    private final int LIMITNUM;

    public WordLimiter(List<String> lines, int limitNum) {
        this.lines = lines;
        this.LIMITNUM = limitNum;
        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        this.wordCounter = new HashMap<>();
        init();
        this.totalWordsCount = wordCounter.keySet().size();
    }

    private void init() {
        for (String line : lines) {
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            tokens.forEach(s -> wordCounter.merge(s, 1, (o, n) -> o + n));
        }
        Map<String, Integer> temp = new LinkedHashMap<>();
        final int[] a = {0};
        if (LIMITNUM == -1) {
            wordCounter.entrySet().parallelStream()
                    .sorted((o1, o2) -> -o1.getValue().compareTo(o2.getValue()))
                    .forEachOrdered(e -> temp.put(e.getKey(), a[0]++));
        }else{
            wordCounter.entrySet().parallelStream()
                    .sorted((o1, o2) -> -o1.getValue().compareTo(o2.getValue()))
                    .limit(LIMITNUM)
                    .forEachOrdered(e -> temp.put(e.getKey(), a[0]++));
        }
        wordCounter = temp;
    }

    public List<String> getLabels() {
        return new ArrayList<>(wordCounter.keySet());
    }

    public int getWordIndex(String word) {
        return wordCounter.getOrDefault(word, wordCounter.get(RnnPredictWords.UNKNOWN));
    }

    public int getTotalWordsCount() {
        return totalWordsCount;
    }

    public String getWord(int index) {
        return wordCounter.entrySet().stream().filter(entry -> entry.getValue() == index)
                .map(Map.Entry::getKey).findFirst().get();
    }

    public String toString() {
        return wordCounter.toString();
    }

    public void toFile(String path) throws IOException {
        List<String> list = new ArrayList<>();
        wordCounter.entrySet().forEach(entry -> list.add(entry.getKey() + "," + entry.getValue()));
        Utils.writeLineToPath(list, path);
    }
}
