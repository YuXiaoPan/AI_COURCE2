package me.peyppicp.method2;

import lombok.Data;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
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
    private WordVectors wordVectors;

    private Map<String, Integer> wordIndexMap;

    public WordToIndex(String samplesFilePath, String wordVectorPath) {
        this.samplesFilePath = samplesFilePath;
        this.wordVectorPath = wordVectorPath;
    }

    private void init() throws IOException {
        samplesFile = new File(samplesFilePath);
        wordVectors = WordVectorSerializer.readWord2VecModel(new File(wordVectorPath));
        wordIndexMap = new HashMap<>();
        List<String> samples = FileUtils.readLines(samplesFile, Charsets.UTF_8);
        samples.parallelStream().forEach(s -> {
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            tokens.parallelStream().filter(token -> wordVectors.hasWord(token)).forEach(token -> {
                int count = wordIndexMap.values().size();
                wordIndexMap.putIfAbsent(token, count + 1);
            });
        });
//        按照value排序
        Map<String, Integer> temp = new HashMap<>();
        wordIndexMap.entrySet()
                .parallelStream().sorted(Map.Entry.comparingByValue())
                .forEachOrdered(entry -> temp.put(entry.getKey(),entry.getValue()));
        wordIndexMap = temp;
    }

    public int getIndex(String word) {
        return wordIndexMap.get(word);
    }

    public void addUnknown() {
        int size = wordIndexMap.values().size();
        wordIndexMap.putIfAbsent(UNKNOWN, ++size);
        wordIndexMap.putIfAbsent(STOP, ++size);
    }

    public int totalIndexNum() {
        return wordIndexMap.values().size();
    }

    public List<String> totalLabels() {
        return wordIndexMap.keySet().parallelStream().collect(Collectors.toList());
    }
}
