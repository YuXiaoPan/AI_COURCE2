package me.peyppicp.cnn;

import me.peyppicp.Utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class WordToIndex {

    private Map<String, Integer> wordToIndexMap;
    private Map<Integer, String> indexToWordMap;
    private int totalWordsCount;
    private final String path;

    public WordToIndex(String path) throws IOException {
        this.path = path;
        List<String> strings = Utils.readLinesFromPath(path);
        this.wordToIndexMap = new LinkedHashMap<>();
        this.indexToWordMap = new LinkedHashMap<>();
        for (String str : strings) {
            String[] split = str.split(",");
            wordToIndexMap.put(split[0], Integer.parseInt(split[1]));
            indexToWordMap.put(Integer.parseInt(split[1]), split[0]);
        }
        this.totalWordsCount = wordToIndexMap.keySet().size();
    }

    public List<String> getLabels() {
        return new ArrayList<>(wordToIndexMap.keySet());
    }

    public int getIndex(String word) {
        return wordToIndexMap.getOrDefault(word, -1);
    }

    public int getTotalWordsCount() {
        return totalWordsCount;
    }

    public String getWord(int index) {
        return indexToWordMap.getOrDefault(index, "<unknown>");
    }

    public String toString() {
        return wordToIndexMap.toString();
    }
}
