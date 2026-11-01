package com.vibetuned.granita.shell

import android.content.res.AssetManager
import org.json.JSONObject

/**
 * The model's sidecar (granitanet.json, written by `granita export --sidecar`): the
 * class list and each class's group. The input spec and the stroke rules it also
 * states are mirrored in code, in StrokeRaster and SymbolContext.
 */
class Sidecar(val model: String, val classes: List<String>, val groups: Map<String, String>) {
    companion object {
        fun load(assets: AssetManager): Sidecar {
            val json = JSONObject(assets.open("granitanet.json").bufferedReader().use { it.readText() })
            val classes = json.getJSONArray("classes").let { array -> List(array.length()) { array.getString(it) } }
            val groups = json.getJSONObject("groups")
            return Sidecar(
                json.getString("model"),
                classes,
                groups.keys().asSequence().associateWith { groups.getString(it) },
            )
        }
    }
}
