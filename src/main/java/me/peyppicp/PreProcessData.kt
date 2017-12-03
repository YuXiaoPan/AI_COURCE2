package me.peyppicp

import com.vdurmont.emoji.EmojiManager
import java.io.File

fun main(args: Array<String>) {
    val file = File("/Users/yuxiao.pan/IdeaProjects/AI-Emoji/src/main/resources/emoji_sample.txt")
    val emojis = EmojiManager.getAll()
    val emojiSet = HashSet<String>()
    val datas = ArrayList<String>()
    emojis.forEach { emojiSet.add(it.unicode) }
    val originData = file.readLines()
    val length = originData.size
    var count = 0;
    originData.forEach { o ->
        run {
            emojiSet.forEach { e ->
                run {
                    if (o.contains(e)) {
                        datas.add(o)
                    }
                }
            }
            println("Count $count is finish. Remain ${length - count}")
            count++
        }
    }
    val targetFile = File("/Users/yuxiao.pan/IdeaProjects/AI-Emoji/src/main/resources/data.txt")
    if (!targetFile.exists()) {
        targetFile.createNewFile()
    } else {
        targetFile.delete()
        targetFile.createNewFile()
    }
    datas.forEach { targetFile.appendText(it + "\n") }
}
