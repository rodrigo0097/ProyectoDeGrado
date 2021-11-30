import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import retro
#Se configura el display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
# Este proyecto se fue realizado útilizando como guía el tutorial de DQN de deep lizard.
# https://www.youtube.com/watch?v=FU-sNVew9ZA
batch_size = 256
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
#que tan frecuente se van a actualizar los target Q values
target_update = 10
memory_size = 100000
learning_rate = 0.001
num_episodes = 10
#lamentablemente mi nvidia gtx 660 de hace 10 años ya no es compatible con torch, tuve que trabajar directamente con el cpu
# Los comentarios los hice exclusivamente en español, este proyecto que enseña a realizar deep lizard en mi opinion tiene
#un gran valor, ojalá pudiera ser compartido este tutorial como futuro material para la clase
#a los que quisieran tomar esta clase en un futuro.
#sadly my old nvidia gtx is no compatible with the newer versions of torch, i had to work using my old core i7
device = torch.device("cpu")
#primero es definida la neural network, para eso extiende nn que es la clase de torch dedicada
#a neural networks


class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        #nn.linear hace referencia a las fully connected layers
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        #Las posibles salidas para este ejemplo de gym son moverse a la derecha o a la izquierda
        #por eso el output es  = 2
        self.out = nn.Linear(in_features=32, out_features=9)
#define el la forward pass fuction para la NN
    def forward(self, t):
        #la imagen es representada como una cadena
        t = t.flatten(start_dim=1)
        #se utiliza relu (rectificador) como función de activacion
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


#tupla que modela la experiencia
Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            #si se alcanza la capacidad máxima se empieza a hacer push de los recuerdos
            #sobre escribiendo las experiencias mas viejas en primer lugar
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    #retorna una muestra de experiencias aleatorias, el set es tan grande como se defina por parametro
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    #comprueba si cuenta con las suficientes experiencias como para generar un sample del tamaño deseado
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


#Esta clase básicamente modela la exploración - explotación con el valor inicial del epsilon
#su valor final y finalmente la tasa de decaimiento que se desea maejar
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    #se calcula la tasa a la que se va a realizar la exploración, con base a el paso actual
    def get_explotarion_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    ##//! vamos a cam
    def select_action(self, state, policy_net):
        rate = strategy.get_explotarion_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            #de tener un epsilon lo suficientemente grande, exploramos
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device)
        else:
        #de lo contrario explotamos lo previamente explorado
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(device)


class CartPoleEnvManager():
    def __init__(self, device):
        #GPU o CPU?
        self.device = device
        #Discrete Actions have been initialized
        self.env = retro.make(game='IceClimber-Nes', use_restricted_actions=retro.Actions.DISCRETE)
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    #retorna el numero de acciones disponibles para un agente dentro del ambiente
    def num_actions_available(self):
        return self.env.action_space.n

    #utilizar step invoca step que ejecuta la acción que entra por parametro, este retorna una tupla
    #compuesta por observaciones de el ambiente, la recompensa (reward), si el episodio terminó (done)
    #E inflormación de dignostico. En terminos practicos solo importa la recompensa y el bool de done
    #_____________________________________________________________________________________________
    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action)
        print(reward)
        print(self.env.unwrapped.buttons)
        print(self.env.action_space.sample())
        return torch.tensor([reward], device=self.device)

    #se verifica si estamos en el estado inicial del episodio
    def just_starting(self):
        return self.current_screen is None

    #retorna el estado actual del ambiente en forma de una imagen procesada de la pantalla
    #un solo estado es representado como la diferencia entre la pantalla actual y la previa
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2-s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    #se renderiza la pantalla como un arreglo de rgb y se traanspoten en el orden de los canales
    #por altura y ancho, esto es estandar en pytorch
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        # screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    # #Se recortan bordes inecesarios que solo "estorban" en el procesamiento
    # def crop_screen(self, screen):
    #     screen_height = screen.shape[1]
    #     top = int(screen_height * 0.4)
    #     bottom = int(screen_height * 0.8)
    #     screen = screen[:, top:bottom, :]
    #     return screen

    def transform_screen_data(self, screen):
        #se guardan todos los valores de manera secuencial en memoria y se definen los pixeles como float32
        #todos los valores son reescalados al dividir por 255
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        #Se convierte screen de un array de numpy a un tensor de Torch
        screen = torch.from_numpy(screen)
        #Compose es una clase de torchvision que nos permite componter transformaciones de imagen, pertmitiendo
        #ejecutar el proceso de manera secuencial seún se especifica
        #se crea una imagen PIL, se reescala y se transforma en tensor
        resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])

        return resize(screen).unsqueeze(0).to(self.device)



def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Duracion')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episodio", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython:
        display.clear_output(wait = True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    #se verifica que la longitud de los valores sea lo suficientemente grande como para calcular el promedio
    #para el periodo dado
    if len(values)  >= period:
        moving_avg = values.unfold(dimension = 0, size = period, step = 1).mean(dim=1).flatten(start_dim=0)
        #.cat concatena la secuencia dada de tensores segun la dimensión especificada
        #se concantena con tensores poblados con 0, esto porque antes de alcanzar el periodo deseado
        #no tiene sentido calcular la media para ese periodo
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)
policy_net = DQN(em.get_screen_height(), em.get_screen_width())
target_net = DQN(em.get_screen_height(), em.get_screen_width())
#se quiere que el sesgo inductivo de la red objetivo sea el mismo que el de la policy
target_net.load_state_dict(policy_net.state_dict())
#la red objetivo se deja en eval mode (solo para inferencia)
target_net.eval()
#se selecciona ADAM (adaptive moment estimation) como el algoritmo de optimización
optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)


#Se extraen los componentes de la experiencia como 4 tensores (estado, acción, siguiente_estado, recompensa)
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t_1 = torch.cat(batch.state)
    t_2 = torch.cat(batch.action)
    t_3 = torch.cat(batch.reward)
    t_4 = torch.cat(batch.next_state)

    return(t_1, t_2, t_3, t_4)


class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    device = torch.device("cpu")
    @staticmethod
    def get_next(target_net, next_states):
        #se asislan los estados finales para no mandarlos a la target network
        #Sabemos que un estado es final porque el máximo de sus Q values sería 0
        #Se representa como true cada uno de los elementos que corresponden a un final state
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        #se con el tensor que se identifico en la line de arriba se hace uno nuevo que identifica
        #todos los estados no finales (solo su ubicación, no valor)
        non_final_state_locations = (final_state_locations == False)
        #se extraen todos los valores de los estados no finales
        non_final_states = next_states[non_final_state_locations]
        #el valor de batch corresponde al la cantidad de next_states almacenada en el tensor next_states
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        #de todos los elementos en el tensor se selecciona el del Q-value mas alto
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

episode_durations = []
for episode in range (num_episodes):

    em.reset()
    state = em.get_state()
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            #es obligatorio hacer 0 el gradiente antes de propagar para que no se acumule
            optimizer.zero_grad()
            loss.backward()
            #actualiza los sesgos
            optimizer.step()
        #si la ejecución es terminada guarda la duración y gráfica el resultado (si es posible)
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    #En vez de actualizar los valores con la propia policy_net
    #se actualizan los objetivos cada que se alcanze el valor  objetivo definido
    #solucionando el problema de ineficiencia que se tenía al "perseguir su propia cola" utilizando solo 1 red
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
em.close()





