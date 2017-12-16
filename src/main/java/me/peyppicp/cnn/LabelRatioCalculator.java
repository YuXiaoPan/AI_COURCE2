package me.peyppicp.cnn;

import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.advance2.WordToIndex;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/16
 * @email yuxiao.pan@kikatech.com
 */
public class LabelRatioCalculator {

    public static void main(String[] args) throws IOException {
        File emojiSampleFile = new File("EmojiSample.txt");
        HashMap<String, Integer> emojiCounterMap = new HashMap<>();
        List<String> emojiSamples = FileUtils.readLines(emojiSampleFile, Charsets.UTF_8);
        for (String emojiSample : emojiSamples) {
            List<String> emojis = EmojiParser.extractEmojis(emojiSample).parallelStream().distinct().collect(Collectors.toList());
            for (String emoji : emojis) {
                emojiCounterMap.merge(emoji, 1, (o, n) -> o + n);
            }
        }
        LinkedHashMap<String, Integer> sortedMap = new LinkedHashMap<>();
        emojiCounterMap.entrySet().parallelStream()
                .sorted((o1, o2) -> -o1.getValue().compareTo(o2.getValue()))
                .limit(24)
                .forEachOrdered(entry -> sortedMap.put(entry.getKey(), entry.getValue()));

        final int[] emojiSum = {0};
        sortedMap.forEach((s, integer) -> emojiSum[0] += integer);

        File file = new File("EmojiSampleLabels.txt");
        HashMap<String, Integer> map = new HashMap<>();
        List<String> labels = FileUtils.readLines(file, Charsets.UTF_8);
        for (String label : labels) {
            String[] split = label.split(",");
            for (String s : split) {
                map.merge(s, 1, (o, n) -> o + n);
            }
        }
        final int[] sum = {0, 0};
        int subSum = 0;
        map.forEach((s, integer) -> sum[0] += integer);
        for (Map.Entry<String, Integer> entry : sortedMap.entrySet()) {
            subSum += entry.getValue();
            System.out.println("label:" + entry.getKey() + ",num:" + entry.getValue() + ",ratio:"
                    + (entry.getValue() / (sum[0] * 1.0d))
                    + ",sumRatio:" + (subSum / (sum[0] * 1.0d)));
        }

        System.out.println("EmojiSum:" + emojiSum[0]);
        System.out.println("Sum:" + sum[0]);
//        Preconditions.checkArgument(emojiSum[0] == sum[0]);

        WordToIndex wordToIndex = new WordToIndex("EmojiSample.txt");
        System.out.println();
    }
}
