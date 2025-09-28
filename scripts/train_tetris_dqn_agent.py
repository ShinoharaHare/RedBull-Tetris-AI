import random
import traceback
from pathlib import Path

import fire
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

from red_bull_tetris_ai.agent import *
from red_bull_tetris_ai.env.tetris import Tetris

Transition = tuple[TetrisDQNState, float, TetrisDQNState, bool]


def main(
    experiment_name: str = 'tetris_dqn_agent',
    save_every_n_episodes: int = 1000,
    batch_size: int = 512,
    memory_size: int = 38400,
    min_memory_size: int = 2560,
    max_tetriminoes: int | None = 350,
    lr: float = 1e-3,
    gradient_clip_val: float | None = None,
    num_episodes: int = 3000,
    eps_min: float = 0.001,
    eps_max: float = 1.0,
    num_decay_episodes: int = 2000,
    num_batch_per_episode: int = 1,
    gamma: float = 0.9999,
    device: torch.device = torch.device('cuda'),
    resume: bool = True,
    resume_optimizer: bool = True,
    resume_memory: bool = True
):
    save_dir = Path('checkpoints') / experiment_name
    save_dir = Path(save_dir)

    model = TetrisDQNAgent()
    model.train().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    memory = Memory[Transition](memory_size)

    checkpoint_paths = list(save_dir.glob('episode-*'))
    if resume and checkpoint_paths:
        episodes = [int(p.stem.removeprefix('episode-')) for p in checkpoint_paths]
        episode, checkpoint_path = max(zip(episodes, checkpoint_paths), key=lambda x: x[0])
        print(f'Resuming from episode {episode} at {checkpoint_path}.')
        
        safetensors.torch.load_model(model, str(checkpoint_path / 'model.safetensors'))

        if resume_optimizer:
            optimizer.load_state_dict(torch.load(checkpoint_path / 'optimizer.pt'))

        if resume_memory:
            memory.load_state_dict(torch.load(checkpoint_path / 'memory.pt'))

    else:
        episode = 0

    env = Tetris(max_tetriminoes=max_tetriminoes)
    progress_bar = tqdm(total=num_episodes, initial=episode, dynamic_ncols=True)
    try:
        while episode < num_episodes:
            env.reset()
            
            state = TetrisDQNState.from_state(env.state)
            total_reward = 0.0
            done = False
            epsilon = eps_min + (eps_max - eps_min) * max(0.0, (num_decay_episodes - episode) / num_decay_episodes)

            while not done:
                next_states = env.get_next_states()
                actions, next_states = zip(*next_states.items())
                next_states = TetrisDQNState.from_batch(next_states)
                next_states = next_states.to(device, non_blocking=True)
                
                if random.random() > epsilon:
                    q_values = model(**next_states.to_dict())[:, 0]
                    index = torch.argmax(q_values).item()

                else:
                    index = random.randrange(len(actions))
                
                action = actions[index]
                next_state = next_states[index]
                reward, done = env.step(action)
                total_reward += reward
                memory.add(
                    (
                        state.to('cpu', non_blocking=True),
                        reward,
                        next_state.to('cpu', non_blocking=True),
                        done
                    )
                )
                state = next_state

            progress_bar.set_postfix(memory=len(memory))

            if len(memory) < min_memory_size:
                continue
            
            for _ in range(num_batch_per_episode):
                transitions = memory.sample(batch_size)
                state_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

                state_batch = TetrisDQNState.from_batch(state_batch)
                state_batch = state_batch.to(device, non_blocking=True)

                reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
                reward_batch = reward_batch.unsqueeze(1)

                next_state_batch = TetrisDQNState.from_batch(next_state_batch)
                next_state_batch = next_state_batch.to(device, non_blocking=True)

                done_batch = torch.tensor(done_batch, dtype=torch.float, device=device)
                done_batch = done_batch.unsqueeze(1)

                q_values = model(**state_batch.to_dict())
                with torch.no_grad():
                    next_q_values = model(**next_state_batch.to_dict())

                target = reward_batch + (1 - done_batch) * gamma * next_q_values            
                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

            episode += 1
            progress_bar.update()

            if episode % save_every_n_episodes == 0:
                sd = save_dir / f'episode-{episode}'
                sd.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_model(model, str(sd / 'model.safetensors'))
                torch.save(memory.state_dict(), sd / 'memory.pt')
                torch.save(optimizer.state_dict(), sd / 'optimizer.pt')

            progress_bar.write(
                f'Episode: {episode} '
                f'Reward: {total_reward:.1f} '
                f'Epsilon: {epsilon:.4f} '
                f'Score: {env.score} '
                f'Combo: {env.max_combo} '
                f'Multiplier: {int(env.score_multiplier)} '
                f'Survived: {env.survived} '
                f'Placed: {env.num_tetriminoes_placed} '
                f'Lines: {env.num_lines_cleared}'
            )

    except KeyboardInterrupt: ...

    except:
        traceback.print_exc()

    finally:
        sd = save_dir / f'episode-{episode}'
        sd.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_model(model, str(sd / 'model.safetensors'))
        torch.save(memory.state_dict(), sd / 'memory.pt')
        torch.save(optimizer.state_dict(), sd / 'optimizer.pt')


if __name__ == '__main__':
    fire.Fire(main)
