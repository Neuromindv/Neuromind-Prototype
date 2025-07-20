<template>
  <div id="app">
    <div id="sidebar">
      <h2>🧠 <span>Kategorie</span></h2>
      <input v-model="search" placeholder="Wyszukaj neuron..." />
      <div v-for="(group, category) in categorizedBranches" :key="category" class="category">
        <strong>{{ category }} ({{ group.length }})</strong>
        <ul>
          <li v-for="item in group" :key="item.id" @click="focusNeuron(item)">
            {{ item.label }}
          </li>
        </ul>
      </div>
    </div>

    <canvas ref="canvas" />

    <div id="question-box">
      <h3>🤖 Pytanie od AI:</h3>
      <p><strong>{{ currentRound.question.text || 'Ładowanie pytania...' }}</strong></p>
      <p>⏳ Czas pozostały: {{ formattedTimeLeft }}</p>

      <div v-for="answer in currentRound.answers" :key="answer.id" class="answer">
        <div><strong>{{ answer.user }}:</strong> {{ answer.content }}</div>
        <button @click="voteAnswer(answer.id)" :disabled="hasVoted(answer.id)">
          🔥 {{ answer.votes }}
        </button>
      </div>

      <div class="answer-input">
        <input v-model="userInput" placeholder="Twoja odpowiedź..." @keyup.enter="submitAnswer" />
        <button @click="submitAnswer">Wyślij</button>
      </div>

      <hr />

      <h4>🧠 Zapytaj AI:</h4>
      <div class="answer-input">
        <input v-model="aiQuestion" placeholder="Twoje pytanie..." @keyup.enter="askAI" />
        <button @click="askAI">Zadaj</button>
      </div>
      <div v-if="aiAnswer" class="ai-response">
        <strong>🧠 Odpowiedź AI:</strong>
        <p>{{ aiAnswer }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

const canvas = ref(null)
const search = ref("")
const userInput = ref("")
const aiQuestion = ref("")
const aiAnswer = ref("")
const selectedNeuron = ref(null)
const allBranches = ref([])
const currentRound = ref({ question: {}, answers: [] })

let scene, camera, renderer, controls, raycaster, mouse
let neuronSpheres = []

const categorizedBranches = computed(() => {
  const filtered = allBranches.value.filter(b => b.label.toLowerCase().includes(search.value.toLowerCase()))
  const grouped = {}
  filtered.forEach(branch => {
    const cat = branch.category || "Inne"
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(branch)
  })
  return grouped
})

const formattedTimeLeft = computed(() => {
  const seconds = currentRound.value.time_left_seconds || 0
  const m = String(Math.floor(seconds / 60)).padStart(2, '0')
  const s = String(seconds % 60).padStart(2, '0')
  return `${m}:${s}`
})

const loadNetworkData = async () => {
  const res = await fetch("http://localhost:8000/network")
  const data = await res.json()
  allBranches.value = data.branches
  renderNetwork(data.branches)
}

const loadCurrentRound = async () => {
  const res = await fetch("http://localhost:8000/round")
  const data = await res.json()
  currentRound.value = data
}

const renderNetwork = (branches) => {
  neuronSpheres.forEach(n => scene.remove(n.mesh))
  neuronSpheres = []

  const coreGeo = new THREE.SphereGeometry(1, 32, 32)
  const coreMat = new THREE.MeshStandardMaterial({ color: 0x800000 })
  const core = new THREE.Mesh(coreGeo, coreMat)
  core.position.set(0, 0, 0)
  scene.add(core)

  const radius = 10
  branches.forEach((branch, i) => {
    const theta = i * 0.3
    const phi = i * 0.15
    const x = radius * Math.sin(phi) * Math.cos(theta)
    const y = radius * Math.sin(phi) * Math.sin(theta)
    const z = radius * Math.cos(phi)

    const color = categoryColor(branch.category)
    const material = new THREE.MeshStandardMaterial({ color })
    const geometry = new THREE.SphereGeometry(0.3, 16, 16)
    const mesh = new THREE.Mesh(geometry, material)

    mesh.position.set(x * 2, y * 2, z * 2)
    mesh.userData.targetPosition = new THREE.Vector3(x, y, z)
    scene.add(mesh)

    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff })
    const points = [core.position.clone(), new THREE.Vector3(x, y, z)]
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMat)
    scene.add(line)

    neuronSpheres.push({ mesh, data: branch, pulseOffset: Math.random() * Math.PI * 2 })
  })
}

const categoryColor = (category) => {
  const map = {
    "AI": 0x00ccff,
    "Emocje": 0xff33aa,
    "Wartości": 0xffff00,
    "Popędy": 0xff6600,
    "Społeczne": 0x66ff66,
    "Egzystencjalne": 0xcc99ff,
    "Funkcje poznawcze": 0xffffff,
  }
  return map[category] || 0x888888
}

const focusNeuron = (branch) => {
  selectedNeuron.value = branch
}

const submitAnswer = async () => {
  const content = userInput.value.trim()
  if (!content) return

  await fetch("http://localhost:8000/answer", {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user: "Anonymous", answer: content })
  })

  userInput.value = ""
  await loadCurrentRound()
  await loadNetworkData() // Odśwież neurony po odpowiedzi
}

const askAI = async () => {
  const question = aiQuestion.value.trim()
  if (!question) return

  aiAnswer.value = "⏳ AI myśli..."
  try {
    const response = await fetch("http://localhost:8000/ask", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    })
    const data = await response.json()
    aiAnswer.value = data.answer || "❌ Brak odpowiedzi."
    await loadNetworkData() // Odśwież neurony po odpowiedzi AI
  } catch (err) {
    aiAnswer.value = "❌ Wystąpił błąd podczas rozmowy z AI."
    console.error(err)
  }

  aiQuestion.value = ""
}

const voteAnswer = async (id) => {
  if (hasVoted(id)) return

  await fetch("http://localhost:8000/vote", {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer_id: id })
  })

  localStorage.setItem(`voted_${id}`, "1")
  await loadCurrentRound()
}

const hasVoted = (id) => {
  return localStorage.getItem(`voted_${id}`) === "1"
}

onMounted(() => {
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
  camera.position.set(0, 0, 15)
  renderer = new THREE.WebGLRenderer({ canvas: canvas.value, antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)

  const ambient = new THREE.AmbientLight(0xffffff, 0.6)
  const point = new THREE.PointLight(0xffffff, 1)
  point.position.set(5, 5, 5)
  scene.add(ambient, point)

  controls = new OrbitControls(camera, renderer.domElement)
  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  const animate = () => {
    requestAnimationFrame(animate)
    controls.update()

    const time = performance.now() * 0.001
    neuronSpheres.forEach(n => {
      const scale = 1 + 0.1 * Math.sin(time + n.pulseOffset)
      n.mesh.scale.set(scale, scale, scale)
      const target = n.mesh.userData.targetPosition
      if (target) n.mesh.position.lerp(target, 0.05)
    })

    renderer.render(scene, camera)
  }

  animate()
  loadNetworkData()
  loadCurrentRound()
})
</script>

<style scoped>
#app {
  margin: 0;
  overflow: hidden;
  display: flex;
  height: 100vh;
}

canvas {
  flex: 1;
  display: block;
  background: black;
}

#sidebar {
  width: 260px;
  background: #1a1a2e;
  color: #fff;
  padding: 20px;
  overflow-y: auto;
  border-right: 2px solid #00ccff;
  flex-shrink: 0;
}

#sidebar h2 {
  color: #ff4477;
}

#sidebar input {
  width: 100%;
  margin-bottom: 10px;
  padding: 6px;
  font-size: 14px;
  background: #222;
  color: white;
  border: 1px solid #444;
}

.category {
  margin-bottom: 20px;
}

.category li {
  cursor: pointer;
  font-size: 14px;
  margin: 4px 0;
  color: #ccc;
}

#question-box {
  position: absolute;
  top: 20px;
  left: 280px;
  background: rgba(0, 0, 0, 0.85);
  color: white;
  padding: 20px;
  border: 1px solid #00ccff;
  border-radius: 8px;
  max-width: 600px;
  z-index: 10;
}

.answer {
  background: #111;
  padding: 10px;
  margin-top: 10px;
  border-radius: 5px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.answer-input {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}

.answer-input input {
  flex: 1;
  padding: 8px;
  background: #222;
  color: white;
  border: 1px solid #444;
}

.answer-input button {
  padding: 8px 12px;
  background: #00ccff;
  color: white;
  border: none;
  cursor: pointer;
}

.ai-response {
  margin-top: 10px;
  padding: 10px;
  background: #222;
  border-left: 4px solid #00ccff;
}
</style>