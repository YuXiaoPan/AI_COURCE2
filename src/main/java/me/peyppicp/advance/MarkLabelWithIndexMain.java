package me.peyppicp.advance;

import com.google.common.base.Preconditions;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MarkLabelWithIndexMain {

    public static void main(String[] args) throws IOException {
        File file = new File("commonLabel.txt");
        List<String> emojis = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> distinctEmoji = emojis.parallelStream().distinct().collect(Collectors.toList());
        ArrayList<String> strings = new ArrayList<>(emojis.size());
        for (String emoji : emojis) {
            if (distinctEmoji.contains(emoji)) {
                strings.add(emoji + "," + distinctEmoji.indexOf(emoji));
            }
        }
        Preconditions.checkState(emojis.size() == strings.size());
        File file1 = new File("commonLabelWithIndex.txt");
        FileUtils.writeLines(file1, Charsets.UTF_8.displayName(), strings, false);
    }
}
