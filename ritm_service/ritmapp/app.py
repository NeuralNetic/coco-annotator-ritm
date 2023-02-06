from aiohttp import web
import asyncio

from aqueduct.integrations.aiohttp import (
    FLOW_NAME,
    AppIntegrator,
)
from .flow import (
    Flow,
    Task,
    get_flow,
)


class RITMView(web.View):
    @property
    def flow(self) -> Flow:
        return self.request.app[FLOW_NAME]

    async def post(self):
        post = await self.request.post()
        json_data_filetream = post.get("points")
        json_data = json_data_filetream.file.read()
        task = Task(json_data=json_data)
        await self.flow.process(task, timeout_sec=15)
        return web.Response(body=task.pred, content_type="image/png")


def prepare_app() -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_post('/predict', RITMView)

    AppIntegrator(app).add_flow(get_flow())

    return app


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    web.run_app(prepare_app(), loop=loop, host='0.0.0.0', port=9019)
