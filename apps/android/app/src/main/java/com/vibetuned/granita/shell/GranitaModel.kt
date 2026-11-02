package com.vibetuned.granita.shell

import android.content.res.AssetManager
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.exp

/**
 * The LiteRT export (granitanet.tflite, an asset), fed the way granita feeds PyTorch:
 * the normalised pixels (1 x 1 x 32 x 32) and the context (1 x 16) in, the logits
 * (1 x 43) out, softmax for the confidences. The tflite keeps litert-torch's names, not
 * the sidecar's: signature `serving_default`, inputs `args_0` (the image) and `args_1`
 * (the context), output `output_0`. LiteRT's CompiledModel API, on the CPU.
 */
class GranitaModel(assets: AssetManager) : AutoCloseable {
    class Top(val label: String, val group: String, val confidence: Float)

    class Prediction(
        val logits: FloatArray,
        val probabilities: FloatArray,
        val top: List<Top>,
        /** The model call alone. */
        val nanos: Long,
    )

    val sidecar = Sidecar.load(assets)
    private val model = CompiledModel.create(assets, "granitanet.tflite", CompiledModel.Options(Accelerator.CPU))
    private val image = model.createInputBuffer(IMAGE, SIGNATURE)
    private val context = model.createInputBuffer(CONTEXT, SIGNATURE)
    private val logits = model.createOutputBuffer(LOGITS, SIGNATURE)

    fun predict(input: FloatArray, ctx: FloatArray, topK: Int = 5): Prediction {
        require(input.size == StrokeRaster.SIZE * StrokeRaster.SIZE && ctx.size == SymbolContext.DIM)
        val start = System.nanoTime()
        image.writeFloat(input)
        context.writeFloat(ctx)
        model.run(mapOf(IMAGE to image, CONTEXT to context), mapOf(LOGITS to logits), SIGNATURE)
        val out = logits.readFloat()
        val nanos = System.nanoTime() - start
        val probabilities = softmax(out)
        val order = probabilities.indices.sortedByDescending { probabilities[it] }.take(topK)
        val top = order.map { i ->
            val label = sidecar.classes[i]
            Top(label, sidecar.groups[label] ?: "other", probabilities[i])
        }
        return Prediction(out, probabilities, top, nanos)
    }

    override fun close() {
        image.close()
        context.close()
        logits.close()
        model.close()
    }

    companion object {
        const val RUNTIME = "2.3.2"
        const val SIGNATURE = "serving_default"
        const val IMAGE = "args_0"
        const val CONTEXT = "args_1"
        const val LOGITS = "output_0"

        fun softmax(x: FloatArray): FloatArray {
            val m = x.max()
            val e = FloatArray(x.size) { exp(x[it] - m) }
            val s = e.sum()
            return FloatArray(x.size) { e[it] / s }
        }
    }
}
