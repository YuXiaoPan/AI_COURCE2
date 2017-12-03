package me.peyppicp

import com.vdurmont.emoji.EmojiManager
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.nio.charset.Charset
import java.util.*

class EmojiDataSetIterator(
        val filePath: String,
        val encoding: Charset,
        val miniBatch: Int,
        val exampleLength: Int,
        val random: Random
) : DataSetIterator {

    var totalChars = 0
    var charToIdsMap = mutableMapOf<Char, Int>() //char 映射
    var charSet = mutableSetOf<Char>()
    var charMap = mutableMapOf<Char, Int>() //char的计数器
    var charsRecord = mutableListOf<Char>() //需要训练的char的总数
    var totalLines: List<String>? = null //总行数
    var emojis = EmojiManager.getAll() //全体emoji
    var exampleStartOffsets: LinkedList<Int> = LinkedList() //存放剩余数据组index信息


    init {
        println("Begin prepare data.")
        val startTime = System.currentTimeMillis()
        totalLines = File(filePath).readLines(encoding)
        val readText = File(filePath).readText(encoding)
//        totalLines?.forEach { s ->
//            kotlin.run {
//                s.toCharArray().forEach { c ->
//                    kotlin.run {
//                        charSet.add(c)
//                        charMap.merge(c, 1, BiFunction { t, u -> t + u })
//                        charsRecord.add(c)
//                    }
//                }
//            }
//        }
        println("totalLines size: ${totalLines?.size}.")
        val defaultCharacterSet = me.peyppicp.getDefaultCharacterSet()
        defaultCharacterSet.forEach { charSet.add(it) }
//        totalLines!!.parallelStream().forEach {
//            kotlin.run {
//                it.toCharArray().forEach { c ->
//                    kotlin.run {
//                        charsRecord.add(c)
//                    }
//                }
//            }
//        }

        charsRecord = readText.toCharArray().filter({ it != null }).filter({ defaultCharacterSet.contains(it) }).toMutableList()
        buildCharIdsMap()
        buildRemainDataIndex()
        println("Over prepare data. Spend ${System.currentTimeMillis() - startTime} ms.")
    }

    fun getMinimalCharacterSet(): CharArray {
        val validChars = LinkedList<Char>()
        run {
            var c = 'a'
            while (c <= 'z') {
                validChars.add(c)
                c++
            }
        }
        run {
            var c = 'A'
            while (c <= 'Z') {
                validChars.add(c)
                c++
            }
        }
        var c = '0'
        while (c <= '9') {
            validChars.add(c)
            c++
        }
        val temp = charArrayOf('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
        for (ch in temp) validChars.add(ch)
        val out = CharArray(validChars.size)
        var i = 0
        for (ch in validChars) out[i++] = ch
        return out
    }

    fun getDefaultCharacterSet(): CharArray {
        val validChars = LinkedList<Char>()
        for (c in getMinimalCharacterSet()) validChars.add(c)
        val additionalChars = charArrayOf('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')
        for (c in additionalChars) validChars.add(c)
        EmojiManager.getAll().forEach { it.unicode.toCharArray().forEach { validChars.add(it) } }
        val out = CharArray(validChars.size)
        var i = 0
        for (c in validChars) out[i++] = c
        return out
    }

    fun buildCharIdsMap(): Unit {
        for ((index, c) in charSet.sorted().withIndex()) {
            charToIdsMap.putIfAbsent(c, index)
        }
//        totalLines?.forEach { line ->
//            kotlin.run {
//                emojis.forEach { emoji ->
//                    kotlin.run {
//                        if (line.contains(emoji.unicode)) {
//                            charToIdsMap.putIfAbsent(emoji.unicode, index)
//                            index++
//                        }
//                    }
//                }
//            }
//        }
    }

    fun buildRemainDataIndex(): Unit {
        val miniBatchesPerEproh = (charsRecord.size) / (exampleLength)
        for (i in 0..miniBatchesPerEproh) {
            exampleStartOffsets.add(i * exampleLength)
        }
        Collections.shuffle(exampleStartOffsets, random)
//        Collections.shuffle(totalLines, random)
//        val total = totalLines!!.size / exampleLength - 1
//        for (i in 0..total) {
//            remainData.add(i * exampleLength)
//        }
    }

    override fun resetSupported(): Boolean = true

    override fun getLabels(): MutableList<String> {
        throw UnsupportedOperationException("Not implemented!")
    }

    override fun cursor(): Int = totalExamples() - exampleStartOffsets.size

    override fun remove() {
        throw UnsupportedOperationException("Not implemented!")
    }

    override fun inputColumns(): Int = charToIdsMap.keys.size

    override fun numExamples(): Int = totalExamples()

    override fun batch(): Int = miniBatch

    override fun next(batchNum: Int): DataSet {
//        if (remainData.size == 0) throw NoSuchElementException()
//        var currentMiniBatchSize = Math.min(batchNum, remainData.size)
//        val input = Nd4j.create(arrayOf(currentMiniBatchSize, charToIdsMap.keys.size, exampleLength).toIntArray(), 'f')
//        val label = Nd4j.create(arrayOf(currentMiniBatchSize, charToIdsMap.keys.size, exampleLength).toIntArray(), 'f')
//        for (i in 0..currentMiniBatchSize) {
//            var startIndex = remainData.removeFirst()
//            var endIndex = startIndex + exampleLength
//            var currentCharIndex = charToIdsMap[charsRecord[i].toString()]
//            var c = 0;
//            for (k in (startIndex + 1)..endIndex) {
//                var nextCharIndex = charToIdsMap[charsRecord[k].toString()]
//                input.putScalar(intArrayOf(i, currentCharIndex!!, c), 1.0)
//                label.putScalar(intArrayOf(i, currentCharIndex, c), 1.0)
//                currentCharIndex = nextCharIndex
//            }
//        }
        val currMinibatchSize = Math.min(batchNum, exampleStartOffsets.size)
        if (exampleStartOffsets.size == 0) throw NoSuchElementException()
        val input = Nd4j.create(intArrayOf(currMinibatchSize, charToIdsMap.keys.size, exampleLength), 'f')
        val label = Nd4j.create(intArrayOf(currMinibatchSize, charToIdsMap.keys.size, exampleLength), 'f')
        for (i in 0..currMinibatchSize) {
            val startIndex = exampleStartOffsets.removeFirst()
            val endIndex = startIndex + exampleLength - 1
            var currentCharIdx = charToIdsMap[charsRecord[startIndex]]!!
            for ((c, j) in ((startIndex + 1)..endIndex).withIndex()) {
                val nextCharIndex = charToIdsMap[charsRecord[j]]!!
                input.putScalar(intArrayOf(i, currentCharIdx, c), 1.0)
                label.putScalar(intArrayOf(i, nextCharIndex, c), 1.0)
                currentCharIdx = nextCharIndex
            }
        }
        return DataSet(input, label)
    }

    override fun next(): DataSet = next(miniBatch)

    override fun totalOutcomes(): Int = charToIdsMap.keys.size

    override fun setPreProcessor(p0: DataSetPreProcessor?) {
        throw UnsupportedOperationException("Not implemented!")
    }

    override fun totalExamples(): Int = totalLines!!.size

    override fun reset() {
        exampleStartOffsets.clear()
        buildRemainDataIndex()
    }

    override fun hasNext(): Boolean = exampleStartOffsets.size > 0

    override fun asyncSupported(): Boolean = true

    override fun getPreProcessor(): DataSetPreProcessor {
        throw UnsupportedOperationException("Not implemented!")
    }
}