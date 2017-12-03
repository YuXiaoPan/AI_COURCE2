package me.peyppicp.advance;

import com.google.common.base.Preconditions;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MarkLabelWithIndexMain2 {

    public static void main(String[] args) throws IOException {
        File file = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\commonLabel2.txt");
        File file1 = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\distinctLines2.txt");
        List<String> emojis = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> totalLines = FileUtils.readLines(file1, Charsets.UTF_8);
        List<String> newTotalLines = new ArrayList<>();
        List<String> newEmojis = new ArrayList<>();

        Preconditions.checkArgument(emojis.size() == totalLines.size());

        for (int i = 0; i < totalLines.size(); i++) {
            String currentLine = totalLines.get(i);
            String currentEmojiLine = emojis.get(i);
            String[] subEmojis = currentEmojiLine.split("\t");
            for (int j = 0; j < subEmojis.length; j++) {
                newTotalLines.add(currentLine);
                newEmojis.add(subEmojis[j]);
            }
        }

        Preconditions.checkArgument(newTotalLines.size() == newEmojis.size());
        ArrayList<String> strings = new ArrayList<>(newEmojis.size());

        List<String> distinctEmoji = newEmojis.parallelStream().distinct().collect(Collectors.toList());

        for (String emoji : newEmojis) {
            if (distinctEmoji.contains(emoji)) {
                strings.add(emoji + "," + distinctEmoji.indexOf(emoji));
            }
        }

        Preconditions.checkState(newTotalLines.size() == strings.size());
        File file2 = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\duplicateLabelWithIndex.txt");
        File file3 = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\duplicateLine.txt");
        FileUtils.writeLines(file3, Charsets.UTF_8.displayName(), newTotalLines, false);
        FileUtils.writeLines(file2, Charsets.UTF_8.displayName(), strings, false);
    }
}
