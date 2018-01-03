package me.peyppicp.ai;

import lombok.Data;
import me.peyppicp.Utils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.IOException;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Data
public class EmojiToIndex {

    public static final String UNKNOWN = "$UNKNOWN";
    public static final String STOP = "$STOP";

    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

    private Map<String, Integer> emojiToIndexMap = new LinkedHashMap<>();
    private Map<Integer, String> indexToEmojiMap = new LinkedHashMap<>();

    public EmojiToIndex() throws IOException {
        Map<String, String> input0 = new LinkedHashMap<>();
        Map<String, String> input1 = new LinkedHashMap<>();
        List<String> pairs = Utils.readLinesFromPath(CNNDecideEmojiMain.PREFIX + "pair_for_emoji.txt");
        for (String pair : pairs) {
            String[] split = pair.split(",");
            input0.put(split[0], split[1]);
            input1.put(split[1], split[0]);
        }

        Map<String, Integer> temp0 = new LinkedHashMap<>();
        Map<Integer, String> temp1 = new LinkedHashMap<>();
        input0.entrySet().stream()
                .sorted(Comparator.comparing(Map.Entry::getValue))
                .forEachOrdered(entry -> temp0.put(entry.getKey(), Integer.parseInt(entry.getValue())));
        input1.entrySet().stream()
                .sorted(Comparator.comparing(Map.Entry::getKey))
                .forEachOrdered(entry -> temp1.put(Integer.parseInt(entry.getKey()),entry.getValue()));
        emojiToIndexMap = temp0;
        indexToEmojiMap = temp1;
    }

    public int getIndex(String word) {
        return emojiToIndexMap.getOrDefault(word, -1);
    }

//    public void addUnknown() {
//        int size = wordIndexMap.values().size();
//        wordIndexMap.putIfAbsent(UNKNOWN, size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
//        wordIndexMap.putIfAbsent(STOP, ++size);
//    }

    public int totalOutputNumber() {
        return emojiToIndexMap.keySet().size();
    }

    public String getEmoji(int index) {
        return indexToEmojiMap.getOrDefault(index, "$UNKNOWN");
    }

    public int totalIndexNum() {
        return emojiToIndexMap.values().size();
    }

    public List<String> totalLabels() {
        return emojiToIndexMap.keySet().parallelStream().collect(Collectors.toList());
    }
}
