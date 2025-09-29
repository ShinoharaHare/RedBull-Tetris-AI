class Controller {
    #commandQueue = [];
    #locked = false;
    #interval = setInterval(this.update.bind(this));
    #commandMap = {
        'touch': this.sendTouchEvent,
        'moveLeft': this.sendMoveLeftEvent,
        'moveRight': this.sendMoveRightEvent,
        'moveDown': this.sendMoveDownEvent,
        'hold': this.sendHoldEvent,
        'hardDrop': this.sendHardDropEvent,
        'hardMoveLeft': this.sendHardMoveLeftEvent,
        'hardMoveRight': this.sendHardMoveRightEvent
    }

    async update() {
        if (this.#locked) {
            return;
        }

        if (this.#commandQueue.length > 0) {
            let { command, x, y } = this.#commandQueue.shift();
            x = x ?? 260;
            y = y ?? 520;

            this.#locked = true;
            await this.#commandMap[command].call(this, x, y);
            this.#locked = false;
        }
    }

    async sendTouchEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        dispatchTouchEvent('touchend', x, y);
    }

    async sendMoveDownEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 80);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 100);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 120);
        await sleep(0);
        dispatchTouchEvent('touchend', x, y);
    }

    async sendMoveLeftEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x - 20, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x - 30, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x - 40, y);
        await sleep(0);
        dispatchTouchEvent('touchend', x - 100, y);
    }

    async sendMoveRightEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x + 20, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x + 30, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x + 40, y);
        await sleep(0);
        dispatchTouchEvent('touchend', x + 100, y);
    }

    async sendHoldEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y - 40);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y - 80);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y - 100);
        await sleep(0);
        dispatchTouchEvent('touchend', x, y - 120);
    }

    async sendHardDropEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 40);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 80);
        await sleep(0);
        dispatchTouchEvent('touchmove', x, y + 100);
        await sleep(0);
        dispatchTouchEvent('touchend', x, y + 120);
    }

    async sendHardMoveLeftEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(1);
        for (let i = 1; i < 10; i++) {
            dispatchTouchEvent('touchmove', x - 60 * i, y);
            await sleep(1);
        }
        dispatchTouchEvent('touchend', x - 600, y);
    }

    async sendHardMoveRightEvent(x, y) {
        dispatchTouchEvent('touchstart', x, y);
        await sleep(1);
        for (let i = 1; i < 10; i++) {
            dispatchTouchEvent('touchmove', x + 60 * i, y);
            await sleep(1);
        }
        dispatchTouchEvent('touchend', x + 600, y);
    }

    enqueueCommand(command) {
        this.#commandQueue.push(command);
    }

    touch(x, y) {
        this.enqueueCommand(
            {
                command: 'touch',
                x: x,
                y: y
            }
        );
    }

    rotate() {
        this.enqueueCommand({command: 'touch'})
    }

    moveLeft() {
        this.enqueueCommand({command: 'moveLeft'});
    }

    moveRight() {
        this.enqueueCommand({command: 'moveRight'});
    }

    moveDown() {
        this.enqueueCommand({command: 'moveDown'});
    }

    hardMoveLeft() {
        this.enqueueCommand({command: 'hardMoveLeft'});
    }

    hardMoveRight() {
        this.enqueueCommand({command: 'hardMoveRight'});
    }

    hold() {
        this.enqueueCommand({command: 'hold'});
    }

    hardDrop() {
        this.enqueueCommand({command: 'hardDrop'});
    }

    hasPendingCommands() {
        return this.#commandQueue.length > 0 || this.#locked;
    }
}


function dispatchTouchEvent(type, x, y) {
    const canvas = document.querySelector('#react-unity-webgl-canvas-1');
    if (canvas === null) {
        return;
    }

    const touch = new Touch({
        identifier: Date.now(),
        target: canvas,
        clientX: x,
        clientY: y,
        radiusX: 11.5,
        radiusY: 11.5,
        rotationAngle: 0,
        force: 1
    });

    const touchEvent = new TouchEvent(
        type,
        {
            cancelable: true,
            bubbles: true,
            changedTouches: [touch],
            targetTouches: type === 'touchend' ? [] : [touch],
            touches: type === 'touchend' ? [] : [touch],
            shiftKey: false
        }
    );

    canvas.dispatchEvent(touchEvent);
}


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


function screenshot() {
    const canvas = document.querySelector('#react-unity-webgl-canvas-1');
    console._log(canvas);
    return canvas.toDataURL();
}


function main() {
    console._log = console.log;

    if ('ontouchstart' in window === false) {
        window.ontouchstart = null;
    }

    document._createElement = document.createElement;
    document.createElement = (type, options) => {
        let el = document._createElement(type, options);
        if (type === 'canvas') {
            el._getContext = el.getContext;
            el.getContext = (contextType, contextAttributes) => {
                if (contextType === 'webgl2' && contextAttributes?.alpha === true) {
                    contextAttributes.preserveDrawingBuffer = true;
                }
                return el._getContext(contextType, contextAttributes);
            }
        }
        return el;
    }

    document.addEventListener('click', e => {
        controller.touch(e.clientX, e.clientY);
    });

    document.addEventListener('keydown', async (e) => {
        if (e.code === 'ArrowUp') {
            controller.rotate();
        }
        else if (e.code === 'KeyC') {
            controller.hold();
        }
        else if (e.code === 'ArrowLeft') {
            if (e.ctrlKey) {
                controller.hardMoveLeft();
            } else{
                controller.moveLeft();
            }
        }
        else if (e.code === 'ArrowRight') {
            if (e.ctrlKey) {
                controller.hardMoveRight();
            } else {
                controller.moveRight();
            }
        }
        else if (e.code === 'ArrowDown') {
            controller.moveDown();
        }
        else if (e.code === 'Space') {
            controller.hardDrop();
        }
    });
}

const controller = new Controller();

main();
