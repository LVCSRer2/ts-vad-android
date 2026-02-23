package com.example.tsvad.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.tsvad.MainViewModel
import com.example.tsvad.model.PersonalVAD

@Composable
fun DetectionScreen(
    viewModel: MainViewModel = viewModel(),
    onReEnroll: () -> Unit = {},
) {
    val isDetecting by viewModel.isDetecting.collectAsState()
    val currentClass by viewModel.currentClass.collectAsState()
    val confidence by viewModel.confidence.collectAsState()
    val recentEvents by viewModel.recentEvents.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Target Speaker Detection",
            style = MaterialTheme.typography.headlineMedium,
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Main detection indicator
        val bgColor by animateColorAsState(
            targetValue = when (currentClass) {
                PersonalVAD.CLASS_TARGET -> Color(0xFF4CAF50)      // green
                PersonalVAD.CLASS_NON_TARGET -> Color(0xFFF44336)  // red
                else -> Color(0xFF9E9E9E)                          // gray
            },
            animationSpec = tween(200),
            label = "bg",
        )

        Box(
            modifier = Modifier
                .size(120.dp)
                .clip(CircleShape)
                .background(bgColor),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = when (currentClass) {
                        PersonalVAD.CLASS_TARGET -> "TARGET"
                        PersonalVAD.CLASS_NON_TARGET -> "OTHER"
                        else -> "SILENCE"
                    },
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${(confidence * 100).toInt()}%",
                    color = Color.White.copy(alpha = 0.8f),
                    fontSize = 14.sp,
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Confidence bars
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            ConfidenceBar("Silence", viewModel.silenceProb.collectAsState().value, Color.Gray)
            ConfidenceBar("Other", viewModel.nonTargetProb.collectAsState().value, Color(0xFFF44336))
            ConfidenceBar("Target", viewModel.targetProb.collectAsState().value, Color(0xFF4CAF50))
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Control button
        Button(
            onClick = {
                if (isDetecting) viewModel.stopDetection() else viewModel.startDetection()
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isDetecting)
                    MaterialTheme.colorScheme.tertiary
                else
                    MaterialTheme.colorScheme.primary,
            ),
        ) {
            Text(
                text = if (isDetecting) "Stop Detection" else "Start Detection",
                fontSize = 18.sp,
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Re-enroll button
        OutlinedButton(
            onClick = onReEnroll,
            enabled = !isDetecting,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Re-enroll Voice")
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Event log
        Text(
            text = "Recent Events",
            style = MaterialTheme.typography.titleSmall,
            modifier = Modifier.align(Alignment.Start),
        )
        Spacer(modifier = Modifier.height(8.dp))

        val listState = rememberLazyListState()
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxWidth()
                .height(140.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
                .padding(8.dp),
        ) {
            items(recentEvents) { event ->
                Text(
                    text = event,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(vertical = 2.dp),
                )
            }
        }
    }
}

@Composable
private fun ConfidenceBar(label: String, value: Float, color: Color) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.width(80.dp),
    ) {
        Box(
            modifier = Modifier
                .width(40.dp)
                .height(100.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Color.LightGray.copy(alpha = 0.3f)),
            contentAlignment = Alignment.BottomCenter,
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(fraction = value.coerceIn(0f, 1f))
                    .background(color),
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = label, fontSize = 12.sp)
    }
}
