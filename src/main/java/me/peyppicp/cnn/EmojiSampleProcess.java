package me.peyppicp.cnn;

import com.google.common.collect.ArrayListMultimap;
import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.advance2.WordToIndex;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class EmojiSampleProcess {

    public static void main(String[] args) throws IOException {
        File file = new File("emoji_sample.txt");
        WordToIndex wordToIndex = new WordToIndex(file.getCanonicalPath());
        List<String> emojiSamples = FileUtils.readLines(file, Charsets.UTF_8);
        Map<String, Integer> wordIndexMap = wordToIndex.getWordIndexMap();
        ArrayListMultimap<String, String> emojiToSamples = ArrayListMultimap.create(); //emoji -> samples
        System.out.println("Begin handle data.");
        int i = 0;
        for (String sample : emojiSamples) {
            List<String> emojis = EmojiParser.extractEmojis(sample);
            if (!emojis.isEmpty()) {
                for (String emoji : emojis) {
                    if (wordIndexMap.containsKey(emoji)) {
                        String s = EmojiParser.removeAllEmojis(sample).trim().toLowerCase();
                        emojiToSamples.put(emoji, s);
                    }
                }
            } else {
                String s = sample.trim().toLowerCase();
                emojiToSamples.put(WordToIndex.STOP, s);
            }
            System.out.println(i++);
        }
        System.out.println("Begin output data to files.");
        File cnn = new File("cnn");
        if (cnn.isDirectory() && cnn.exists()) {
            cnn.delete();
        }
        cnn.mkdir();
        for (String emoji : emojiToSamples.keySet()) {
            File emojiFile = new File(cnn, emoji + ".txt");
            FileUtils.writeLines(emojiFile,
                    "UTF-8",
                    emojiToSamples.get(emoji).parallelStream().distinct().collect(Collectors.toList()),
                    "\n",
                    false);
        }
    }
}
