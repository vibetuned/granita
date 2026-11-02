package com.vibetuned.granita.shell

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

/**
 * A reply saved for the dataset: musicset's contribution record (`musicset.strokes`:
 * `label`, `writer`, `staff_space`, `strokes` with time and pressure, `context`, `device`,
 * `app`, `time`) and what else the shell knows — the staff the drawing was made on, the
 * notes placed on it, the fraction it was drawn at, and the model's answer.
 */
class Reply(
    val label: String,
    val writer: String,
    val staffSpace: Double,
    /** One list per pen-down: `[x, y, t_ms, pressure]`, or `[x, y, t_ms]` from a finger. */
    val strokes: List<List<DoubleArray>>,
    val context: SymbolContext,
    val staff: Staff,
    val fraction: Double,
    val predicted: GranitaModel.Top?,
    val device: String,
    val app: String,
    val time: String,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("label", label)
        put("writer", writer)
        put("staff_space", staffSpace)
        put("strokes", JSONArray().apply { for (s in strokes) put(JSONArray().apply { for (p in s) put(JSONArray().apply { for (v in p) put(v) }) }) })
        put("context", JSONObject().apply {
            put("y_staff", context.yStaff)
            put("x_measure", context.xMeasure)
            put("barline", context.barline)
            put("note_dx", context.noteDx ?: JSONObject.NULL)
            put("note_dy", context.noteDy ?: JSONObject.NULL)
            put("note_w", context.noteW ?: JSONObject.NULL)
            put("note_h", context.noteH ?: JSONObject.NULL)
            put("width", context.width)
            put("height", context.height)
        })
        put("staff", JSONObject().apply {
            put("space", staff.space)
            put("middle_y", staff.middleY)
            put("left_x", staff.leftX)
            put("right_x", staff.rightX)
        })
        put("notes", JSONArray().apply { for (n in staff.notes) put(JSONArray().apply { put(n.x.toDouble()); put(n.y.toDouble()) }) })
        put("fraction", fraction)
        put("predicted", predicted?.let { JSONObject().apply { put("label", it.label); put("confidence", it.confidence.toDouble()) } } ?: JSONObject.NULL)
        put("device", device)
        put("app", app)
        put("time", time)
    }

    companion object {
        private fun JSONObject.optDoubleOrNull(key: String): Double? = if (isNull(key)) null else getDouble(key)

        fun fromJson(j: JSONObject): Reply {
            val c = j.getJSONObject("context")
            val s = j.getJSONObject("staff")
            val notes = j.optJSONArray("notes")
            val strokes = j.getJSONArray("strokes").let { ss ->
                List(ss.length()) { i -> ss.getJSONArray(i).let { st -> List(st.length()) { k -> st.getJSONArray(k).let { p -> DoubleArray(p.length()) { p.getDouble(it) } } } } }
            }
            return Reply(
                j.getString("label"),
                j.getString("writer"),
                j.getDouble("staff_space"),
                strokes,
                SymbolContext(c.getDouble("y_staff"), c.getDouble("x_measure"), c.getDouble("barline"), c.optDoubleOrNull("note_dx"), c.optDoubleOrNull("note_dy"), c.getDouble("width"), c.getDouble("height"), c.optDoubleOrNull("note_w"), c.optDoubleOrNull("note_h")),
                Staff(s.getDouble("space"), s.getDouble("middle_y"), s.getDouble("left_x"), s.getDouble("right_x"), notes?.let { ns -> List(ns.length()) { i -> ns.getJSONArray(i).let { android.graphics.PointF(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) } } } ?: emptyList()),
                j.getDouble("fraction"),
                j.optJSONObject("predicted")?.let { GranitaModel.Top(it.getString("label"), "", it.getDouble("confidence").toFloat()) },
                j.optString("device", ""),
                j.optString("app", ""),
                j.optString("time", ""),
            )
        }
    }
}

/**
 * The replies on this device: `replies.jsonl` in the app's external files (a file manager
 * shows it under Android/data; the Share button sends it), one record a line, appended as
 * they come. The writer id is this install's, a UUID kept in the preferences.
 */
class Annotations(context: Context, directory: File? = null) {
    val file: File = File(directory ?: context.getExternalFilesDir(null) ?: context.filesDir, FILE_NAME)
    val writer: String
    var count: Int = if (file.isFile) file.readLines().count { it.isNotBlank() } else 0
        private set
    private val appContext = context.applicationContext

    init {
        val prefs = context.getSharedPreferences("granita", Context.MODE_PRIVATE)
        writer = prefs.getString("writer", null) ?: UUID.randomUUID().toString().lowercase(Locale.ROOT).also { prefs.edit().putString("writer", it).apply() }
    }

    fun append(reply: Reply) {
        file.appendText(reply.toJson().toString() + "\n")
        count++
    }

    /** Every reply in the file, in order; a line that does not parse is skipped. */
    fun all(): List<Reply> = if (file.isFile) file.readLines().filter { it.isNotBlank() }.mapNotNull { try { Reply.fromJson(JSONObject(it)) } catch (e: Exception) { null } } else emptyList()

    /** The file, handed to another app. */
    fun shareIntent(): Intent {
        val uri = FileProvider.getUriForFile(appContext, appContext.packageName + ".files", file)
        return Intent.createChooser(
            Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, FILE_NAME)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            },
            "Share the replies",
        )
    }

    companion object {
        const val FILE_NAME = "replies.jsonl"

        fun now(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(Date())
    }
}
